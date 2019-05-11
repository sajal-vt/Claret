from __future__ import absolute_import, print_function
import numpy as np
import pyopencl as cl


def test():
	context = cl.create_some_context()
	queue = cl.CommandQueue(context)
	mf = cl.mem_flags

	a_np = np.random.rand(50000).astype(np.float32)
	b_np = np.random.rand(50000).astype(np.float32)

	a_g = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a_np)
	b_g = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b_np)

	prg = cl.Program(context, """
		__kernel void sum(
		    __global const float *a_g, __global const float *b_g, __global float *res_g)
		{
		  int gid = get_global_id(0);
		  res_g[gid] = a_g[gid] + b_g[gid];
		}
		""").build()

	res_g = cl.Buffer(context, mf.WRITE_ONLY, a_np.nbytes)
	prg.sum(queue, a_np.shape, None, a_g, b_g, res_g)

	res_np = np.empty_like(a_np)
	cl.enqueue_copy(queue, res_np, res_g)

	# Check on CPU with Numpy:
	print(res_np - (a_np + b_np))
	print(np.linalg.norm(res_np - (a_np + b_np)))
	assert np.allclose(res_np, a_np + b_np)

test()