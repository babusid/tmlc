Some examples to figure out syntax / api...

class Add(TensorOp):
    ... 
    def emit_ir(self, inputs: list[ComputeTensor]) -> ComputeProgram:
        assert len(inputs) == 2
        assert all(...) # check shape equiv
        # declare iteration axes on the tensors
        left = inputs[0]
        right = inputs[1]
        axes = (Axis(type=SPATIAL_AXIS, extent=shape) for shape in left.shape)
        lhs = ScalarOperand(tensor=left, axes=axes)
        rhs = ScalarOperand(tensor=right, axes=axes)
        body = (lhs + rhs)
        retblock = ComputeBlock(..., body=body)
        program = ComputeProgram() # frozen dataclass
        program.blocks.append(retblock) # add the block to the program
        program.tensors.extend(inputs) # add the input blocks to the program
        program.tensors.extend(retblock.tensor) # add the output of the block to the program
        return program

class Add(TensorOp):
    ... 
    @override
    def emit_ir(self, inputs: list[ComputeTensor]) -> ComputeProgram:
        assert len(inputs) == 2
        assert all(...) # check shape equiv
        # declare iteration axes on the tensors
        left = inputs[0]
        right = inputs[1]
        axes = (Axis(type=SPATIAL_AXIS, extent=shape) for shape in left.shape)
        body = left[axes] + right[axes]
        retblock = ComputeBlock(..., body=body)
        program = ComputeProgram(blocks=[retblock], tensors=[*inputs, retblock.tensor]) # frozen dataclass
        return program

class 2dTranspose(TensorOp):
    ... 
    @override
    def emit_ir(self, inputs: list[ComputeTensor]) -> ComputeProgram:
        inp = inputs[0]
        retshape = inp.shape[::-1]
        assert len(inp.shape) == 2
        axes = (Axis(type=SPATIAL_AXIS, extent=shape) for shape in inp.shape)
        retaxes = (Axis(type=SPATIAL_AXIS, extent=shape) for shape in retshape)
        ret_tensor = ComputeTensor(shape=retshape, name="2dtranspose", dtype=inp.dtype)
        # overload setitem on ComputeTensor to be a ScalarExpr Assign? that sets the two equal?
        body = (ret_tensor[retaxes] = inp[axes]) 
        retblock = ComputeBlock(..., body=body)
        program = ComputeProgram(blocks=[retblock], tensors=[*inputs, retblock.tensor]) # frozen dataclass
        return program

- Declaratively creates Axis objects.
- Axis dataclass holds a type and an extent. 
- Can declaratively create a `ScalarOperand` which represents an binding of an Axis collection and a ComputeTensor.
- Indexing a ComputeTensor with axis should be a `ScalarOperand` creation sugar (ie. returns one of them)
- `ScalarOperands` are used in a `ScalarExpr` as leaves 
- `ScalarExpr` represents the body of a ComputeBlock
- A `ComputeBlock` produces a single ComputeTensor.
- How do we handle temporary tensors / chain blocks together? we need a way to get the output ComputeTensor defined by a block...
- `ComputeBlock` should implement a `call` interface that returns a `ComputeTensor`. The shape of the `ComputeTensor` can be inferred,
   via the `ScalarOperands` and the `ScalarExpr` in the `ComputeBlock` body. 
- `ComputeBlock` has a method / property `.tensor` that gets a `ComputeTensor` that contains its own name/dtype/shape based on the
  `ComputeBlock`s' `ScalarExpr` body.
- `ComputeBlock`s and tensors get housed inside a `ComputeProgram`. Blocks get executed sequentially (?)
- `ComputePrograms` are chainable / compositional
- We can probably add some builders for handwriting sugar later (ie. what if you need to define a block to get a tmp tensor and then do
  something with that one? Current syntax would probably be kind of annoying)
