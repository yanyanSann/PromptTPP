import os


def main():
    # one can find this dir in the config out file
    log_dir = 'CHECKPOINTS_DIR'
    os.system('tensorboard --logdir={}'.format(log_dir))
    return


if __name__ == '__main__':
    main()
