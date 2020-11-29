from imageai.Detection import ObjectDetection
import os

# Program ini diberikan sebuah gambar, selanjutnya
# Program ini akan memberikan gambar yang telah diberikan keterangan objek pada gambar tersebut

input('Pastikan gambar yang ingin anda deteksi bernama "image.jpg", jika sudah benar klik ENTER')

execution_path = os.getcwd()
detector = ObjectDetection()

detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()

detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "image.jpg"),
                                             output_image_path=os.path.join(execution_path , "imagenew.jpg"))

for eachObject in detections:
    print(eachObject["name"] , " : " , eachObject["percentage_probability"] )

input('Proses berhasil, nama gambar yang sudah dideteksi akan bernama "imagenew.jpg"')