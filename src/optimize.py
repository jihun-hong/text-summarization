import argparse
import torch
import os


def crop(path, name):

    list = path.split("/")
    file = "/".join(list[:-1])

    return os.path.join(file, name)


def shrink_model(args):

    path = args.path
    name = args.new_name
    new_path = crop(path, name)

    if path == '' or name == '':
        print("Error: The value of path or file name is null")
        return

    pt = torch.load(path, map_location=lambda storage, loc: storage)
    model_state = pt['model']
    small_model = {'model': model_state}
    if not os.path.exists(new_path):
        torch.save(small_model, new_path)

    print("Congrats! Model shrinking was successful")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("-path", default='', type=str)
    parser.add_argument("-new_name", default='', type=str)

    args = parser.parse_args()
    shrink_model(args)
