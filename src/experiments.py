from Depth-Anything-V2.depth_anything

ENSEMBLE_SIZE = 4
saved_model_paths = ["/kaggle/input/cil-model-ensemble/model2025-05-05-13-14-18", 
                     "/kaggle/input/cil-model-ensemble/model2025-05-06-12-36-04",
                     "/kaggle/input/cil-model-ensemble/model2025-05-06-15-08-10",
                     "/kaggle/input/cil-model-ensemble/model2025-05-06-17-40-46"
                    ]
var_model_path = "/kaggle/input/cil-var-model/variance-model2025-05-06-21-43-21"
models = []
for i in range(ENSEMBLE_SIZE):
    model = DepthAnythingV2(**{**model_configs[model_encoder], 'max_depth': max_depth}).to('cuda')
    model.load_state_dict(torch.load(saved_model_paths[i]))
    models.append(model)
var_model = DepthAnythingV2(**{**model_configs[model_encoder], 'max_depth': max_depth}).to('cuda')
var_model.load_state_dict(torch.load(var_model_path))


# Compute AUSE and AURG of output variance
def experiment_1():
    pass

# Run evaluation on chosen model
def experiment_2():
    pass

# Read Kalman Gain
def experiment_3():
    pass