import splitfolders

if __name__ == '__main__':
    splitfolders.ratio(r'C:\DogBreed\Images', output=r"C:\DogBreed\splited_image", seed=1337, ratio=(0.8,0.1,0.1))