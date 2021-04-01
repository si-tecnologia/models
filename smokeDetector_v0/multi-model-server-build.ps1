docker rm pantera-model -f
docker run -dt -p 8501:8501 `
    --mount type=bind,source="$(pwd)"/fine_tuned_model/saved_model,target=/models/effdet_v0/1/ `
    --mount type=bind,source="$(pwd)"/fine_tuned_model/saved_model_v4,target=/models/effdet_v4/1/ `
    --mount type=bind,source="$(pwd)"/tfxConfig/models.config,target=/tfxConfig/models.config `
    --name=pantera-model `
    --restart=always `
    -t tensorflow/serving `
    --model_config_file=/tfxConfig/models.config
