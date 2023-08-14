class Parameter():
    def __init__(self):
        self.root = '/home/topnet/图片'
        self.folders = ['head2', 'tail', 'other']
        self.img_size = 320
        self.output_shape = 3
        self.tensorflow_backend = True
        self.svm_backend = False
        self.faltten = True
        self.use_img = False
        self.use_text = True
        self.show_res = False
