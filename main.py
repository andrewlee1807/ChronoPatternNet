#  Copyright (c) 2024 Andrew
#  Email: andrewlee1807@gmail.com
#
import argparse
import os

from models import get_model, build_callbacks
from utils.data import Dataset, ChronoGenerator, TimeSeriesGenerator
from utils.logging import arg_parse, warming_up, close_logging
import time


def main():
    parser = argparse.ArgumentParser()
    args = arg_parse(parser)

    config = warming_up(args)

    # Load dataset
    dataset = Dataset(dataset_name=config["dataset_name"])

    data = dataset.dataloader.export_the_sequence(config["features"])

    print("Building time series generator...")
    if "Chrono".upper() in args.model_name.upper():
        tsf = ChronoGenerator(data=data,
                              config=config,
                              normalize_type=1,
                              shuffle=False)
    else:
        tsf = TimeSeriesGenerator(data=data,
                                  config=config,
                                  normalize_type=1,
                                  shuffle=False)

    data_train = [tsf.data_train[0], tsf.data_train[1]]
    data_valid = [tsf.data_valid[0], tsf.data_valid[1]]
    data_test = [tsf.data_test[0], tsf.data_test[1]]

    print("Building model...")
    # Get model (built and summary)
    model = get_model(model_name=args.model_name,
                      config=config)

    # callbacks
    callbacks = build_callbacks(tensorboard_log_dir=config["tensorboard_log_dir"])

    # Train model
    t1 = time.time()
    history = model.fit(x=data_train[0],  # [number_recoder, input_len, number_feature]
                        y=data_train[1],  # [number_recoder, output_len, number_feature]
                        validation_data=data_valid,
                        epochs=config["epochs"],
                        callbacks=[callbacks],
                        verbose=2,
                        batch_size=64,
                        use_multiprocessing=True)
    t2 = time.time()
    print("=============================================================")
    print("Minimum val mse:")
    print(min(history.history['val_mse']))
    print("Minimum training mse:")
    print(min(history.history['mse']))
    result = model.evaluate(data_test[0], data_test[1], batch_size=1,
                            verbose=2,
                            use_multiprocessing=True)
    t3 = time.time()
    print("Evaluation result: ")
    print(config['output_length'], result[1], result[2])
    print("Time training: ", t2 - t1)
    time_record = f'{os.path.join(config["output_dir"], config["dataset_name"])}_time_executing.txt'
    file = open(time_record, 'a')
    file.write(f'{config["output_length"]},{t2 - t1},{t3 - t2}\n')
    file.close()

    result_file = f'{os.path.join(config["output_dir"], config["dataset_name"])}_evaluation_result.txt'
    file = open(result_file, 'a')
    file.write(f'{config["output_length"]},{result[1]},{result[2]}\n')
    file.close()

    if args.write_log_file:
        close_logging(config["file"], config["orig_stdout"])


if __name__ == '__main__':
    main()
