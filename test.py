import tensorflow as tf
import time 
import numpy as np 

def wrap_frozen_graph(graph_def, inputs = None, outputs=None, print_graph=False):
    def _imports_graph_def():
        tf.compat.v1.import_graph_def(graph_def, name="")

    wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])
    import_graph = wrapped_import.graph

    print("-" * 50)
    print("Frozen model layers: ")
    layers = [op.name for op in import_graph.get_operations()]
    if print_graph == True:
        for layer in layers:
            print(layer)
    print("-" * 50)
    return wrapped_import.prune(
        tf.nest.map_structure(import_graph.as_graph_element, inputs),
        tf.nest.map_structure(import_graph.as_graph_element, outputs))


def main():
    start_time = time.time()
    # Load frozen graph using TensorFlow 1.x functions
    with tf.io.gfile.GFile("ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb", "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        loaded = graph_def.ParseFromString(f.read())

    # Wrap frozen graph to ConcreteFunctions
    frozen_func = wrap_frozen_graph(graph_def=graph_def,
                                    print_graph=True)
    end_time = time.time()
    print ("--- %s seconds ---" % (end_time - stat_time))


if __name__ == '__main__':
    main()

    
    unsupported = [l for l in self.network.layers.keys() if l not in supported]
        if len(unsupported) != 0:
            print("Unsupported layers found: {}".format(unsupported))
            print("Check whether extensions are available to add to IECore.")
            exit(1)