docker rm pantera-model -f
docker run -dt -p 8501:8501 `
    --mount type=bind,source="$(pwd)"/fine_tuned_model_versioned/,target=/models/effdet `
    --mount type=bind,source="$(pwd)"/tfxConfig/models2.config,target=/tfxConfig/models2.config `
    --name=pantera-model `
    --restart=always `
    -t tensorflow/serving `
    --model_config_file=/tfxConfig/models2.config
