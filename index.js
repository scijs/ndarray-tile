'use strict'

var pool = require('ndarray-scratch')
var ops = require('ndarray-ops')

module.exports = ndarrayTile

function ndarrayTile () {
  var i, input, output, reps

  if (arguments.length === 2) {
    // With allocation (output not specified):
    input = arguments[0]
    reps = arguments[1]
  } else if (arguments.length === 3) {
    // Without allocation (output specified):
    output = arguments[0]
    input = arguments[1]
    reps = arguments[2]
  }

  if (!Array.isArray(reps)) {
    throw new Error('second argument of tile must be an array of repetition counts for each dimension')
  }

  // Calculate the output dimensions:
  var inputShape = input.shape.slice(0)
  var newShape = inputShape.slice(0)
  var newDim = Math.max(newShape.length, reps.length)
  for (i = 0; i < newDim; i++) {
    inputShape[i] = inputShape[i] === undefined ? 1 : inputShape[i]
    reps[i] = reps[i] === undefined ? 1 : reps[i]
    newShape[i] = (newShape[i] === undefined ? 1 : newShape[i]) * reps[i]

    if (newShape[i] === 0) {
      throw new Error('Number of tiles must be greater than zero')
    }
  }

  // Allocate output:
  if (!output) {
    output = pool.zeros(newShape, input.dtype)
  }

  var repProd = 1
  for (i = 0; i < reps.length; i++) {
    repProd *= reps[i]
  }

  var origin = new Array(reps.length)
  for (i = 0; i < reps.length; i++) {
    origin[i] = 0
  }

  for (i = 0; i < repProd; i++) {
    var toPick = newShape.slice(0)
    for (d = 0; d < newShape.length; d++) {
      toPick[d] = d < input.dimension ? null : origin[d]
    }

    var slice = output.pick.apply(output, toPick)
    slice = slice.lo.apply(slice, origin)
    slice = slice.hi.apply(slice, input.shape)

    ops.assign(slice, input)

    // Skip index update on last iteration:
    if (i === repProd - 1) break

    var d = newDim - 1
    while (d >= 0) {
      origin[d] += inputShape[d]
      if (origin[d] === newShape[d]) {
        origin[d] = 0
        d--
      } else break
    }
  }

  return output
}
