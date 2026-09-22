"""
Reduction-free inlining: the one pass that subsumes both elementwise fusion and shape collapse.

A ComputeBlock whose body performs no reduction computes each output element as an independent
function of its inputs. Such a block can be *inlined* into its consumers: wherever a consumer reads
the producer's output at some coordinate, we splice in the producer's body with the producer's axes
substituted by that coordinate. Because the substitution composes the producer's affine index maps
with the consumer's, this single mechanism covers:

* elementwise fusion -- ``((x + y) * z) + y`` collapses from three blocks to one; and
* shape-manipulation collapse -- a ``reshape -> broadcast -> transpose`` chain (each lowered to a
  reduction-free block whose body is a single remapped ``Read``) folds into one composed ``Read``.

Both are the hand-written optimizations in ``tests/compute_fusion.py``, now done mechanically.

What is *never* inlined (hard legality constraints, independent of any cost model):
* blocks with a reduction (``reduce_axes``/``combiner``) -- inlining would force the consumer to run
  the reduction inside its own domain, which is not a per-element substitution; and
* program outputs -- they must be materialized so the runtime can hand them back.

A producer's *inputs* are still fused into a reduction block's body (prologue fusion), and a
reduction block's output is still fused into downstream reduction-free consumers (epilogue fusion);
only the reduction block itself stays materialized.

Whether a *legal* producer is *worth* inlining is the ``should_inline`` seam. The default says "yes"
universally (matching the current goal of always inlining shape/elementwise ops), but it receives an
``InlineCandidate`` carrying everything a real cost model would weigh -- number of consumers, number
of distinct read sites (the recompute/duplication factor), body size -- so swapping in a cost model
later is a one-function change. Note the interplay with memory planning: every inlined producer is a
ComputeTensor that disappears from the program, so downstream bufferization simply never allocates a
buffer for it -- fusion *is* the first, coarsest form of memory planning.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from typing_extensions import override

from tmlc.compute.program import ComputeBlock, ComputeProgram, ComputeTensor
from tmlc.compute.rewrite import iter_reads, replace_reads, subst_body
from tmlc.compute.scalar import Read, ScalarExprBase
from tmlc.compute.transform import ComputeProgramTransform


@dataclass(frozen=True)
class InlineCandidate:
    """
    Everything the fusion decider sees about a reduction-free producer.

    ``read_sites`` counts distinct ``Read`` occurrences across all consumer bodies (a producer read
    twice in one consumer counts twice); it is the factor by which the producer's work is duplicated
    if inlined, and is exactly what a cost model trades off against the memory traffic saved.
    """

    producer: ComputeBlock
    consumers: tuple[ComputeBlock, ...]
    read_sites: int
    is_output: bool
    has_reduce: bool

    @property
    def body_size(self) -> int:
        """Node count of the producer body -- a crude proxy for recompute cost."""
        return len(iter_reads(self.producer.body))


def default_should_inline(candidate: InlineCandidate) -> bool:
    """Universally inline (matches "always fuse shape/elementwise ops for now")."""
    return True


ShouldInline = Callable[[InlineCandidate], bool]


class InlineReductionFree(ComputeProgramTransform):
    """
    Inline every legal reduction-free block the ``should_inline`` hook approves.

    Blocks are visited in the program's existing dependency order, so by the time a block is
    reached every inlinable producer feeding it already has a fully-inlined body recorded; splicing
    is then a single level of substitution and never needs to recurse through further producers.
    """

    def __init__(self, should_inline: ShouldInline = default_should_inline) -> None:
        self._should_inline = should_inline

    @override
    def __call__(self, program: ComputeProgram) -> ComputeProgram:
        output_set = set(program.outputs)  # ComputeTensor is identity-eq
        consumers, read_sites = self._consumer_index(program)

        # output tensor -> fully-inlined body, parameterized by the producer's own output_axes.
        inlined: dict[ComputeTensor, ScalarExprBase] = {}
        # output tensor -> producer block, so an inlined read site knows which axes to bind.
        producer_of: dict[ComputeTensor, ComputeBlock] = {}
        kept_blocks: list[ComputeBlock] = []

        for block in program.blocks:
            body = self._inline_reads(block.body, inlined, producer_of)

            has_reduce = bool(block.reduce_axes)
            is_output = block.output in output_set
            legal = not has_reduce and not is_output
            candidate = InlineCandidate(
                producer=block,
                consumers=tuple(consumers.get(block.output, ())),
                read_sites=read_sites.get(block.output, 0),
                is_output=is_output,
                has_reduce=has_reduce,
            )

            if legal and self._should_inline(candidate):
                inlined[block.output] = body
                producer_of[block.output] = block
            else:
                kept_blocks.append(
                    ComputeBlock(
                        output=block.output,
                        output_axes=block.output_axes,
                        reduce_axes=block.reduce_axes,
                        body=body,
                        combiner=block.combiner,
                    )
                )

        return self._rebuild(program, kept_blocks)

    def _inline_reads(
        self,
        body: ScalarExprBase,
        inlined: dict[ComputeTensor, ScalarExprBase],
        producer_of: dict[ComputeTensor, ComputeBlock],
    ) -> ScalarExprBase:
        """Splice each read of an inlined producer with its body at the read's coordinates."""

        def replace(read: Read) -> ScalarExprBase | None:
            producer_body = inlined.get(read.tensor)
            if producer_body is None:
                return None
            producer = producer_of[read.tensor]
            # bind the producer's output axes to the coordinates this consumer read them at
            mapping = dict(zip(producer.output_axes, read.index))
            return subst_body(producer_body, mapping)

        return replace_reads(body, replace)

    @staticmethod
    def _consumer_index(
        program: ComputeProgram,
    ) -> tuple[dict[ComputeTensor, list[ComputeBlock]], dict[ComputeTensor, int]]:
        """Map each tensor to the blocks that read it, and to its total distinct read-site count."""
        consumers: dict[ComputeTensor, list[ComputeBlock]] = {}
        read_sites: dict[ComputeTensor, int] = {}
        for block in program.blocks:
            reads = iter_reads(block.body)
            seen: set[ComputeTensor] = set()
            for read in reads:
                read_sites[read.tensor] = read_sites.get(read.tensor, 0) + 1
                if read.tensor not in seen:
                    consumers.setdefault(read.tensor, []).append(block)
                    seen.add(read.tensor)
        return consumers, read_sites

    @staticmethod
    def _rebuild(program: ComputeProgram, kept_blocks: list[ComputeBlock]) -> ComputeProgram:
        """Reassemble the program, dropping the tensors of every inlined (now absent) block."""
        survivors = set(program.inputs)
        survivors |= {tensor for tensor, _ in program.constants}
        survivors |= {block.output for block in kept_blocks}
        tensors = tuple(t for t in program.tensors if t in survivors)
        return ComputeProgram(
            tensors=tensors,
            blocks=tuple(kept_blocks),
            inputs=program.inputs,
            outputs=program.outputs,
            constants=program.constants,
        )
