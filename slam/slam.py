import pathlib

def load_images():
    pass


def main():
    folder = pathlib.Path('../dataset/seq50rect3')
    images = load_images(folder)


if __name__ == "__main__":
    main()