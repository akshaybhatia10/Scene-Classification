import tensorflow_hub as hub
import tensorflow as tf

def build_graph(hub_module):
	"""
	Build a graph from tensorflow hub module
	Args:
		hub_module: Tensorflow Hub module
	Returns:
		graph extracted from hub module
		bottleneck tensor
		input tensor (expected image size by graph)	

	"""
	module = hub.load_module_spec(hub_module)
	h, w = hub.get_expected_image_size(module)
	with tf.Graph() as graph:
		input_tensor = tf.placeholder(tf.float32, shape=(None, h, w, 3))
		mod = hub.Module(module)
		final_tensor = m(input_tensor)
		do_quant = any(for op in graph.as_graph_def())


	return graph, final_tensor, input_tensor	

