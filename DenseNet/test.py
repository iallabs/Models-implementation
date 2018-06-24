from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

print_tensors_in_checkpoint_file("ckpt/tf-densenet121.ckpt", tensor_name="", all_tensors="", all_tensor_names=True)