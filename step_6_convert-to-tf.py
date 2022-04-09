import onnx
from onnx_tf.backend import prepare

onnx_model = onnx.load("signlanguage.onnx")
tf_rep = prepare(onnx_model)
tf_rep.export_graph("signlanguage_tfmodel.pb")