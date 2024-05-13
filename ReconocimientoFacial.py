import tkinter as tk
import cv2
import os
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

def iniciar_entrenamiento():
    nombre = nombre_entry.get()
    dir_faces = r'C:\Users\ASUS\Desktop\cara'
    path = os.path.join(dir_faces, nombre)
    size = 4

    if not os.path.isdir(path):
        os.mkdir(path)

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)

    img_width, img_height = 112, 92

    count = 0
    while count < 100:
        rval, img = cap.read()
        img = cv2.flip(img, 1, 0)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mini = cv2.resize(gray, (int(gray.shape[1] / size), int(gray.shape[0] / size)))

        faces = face_cascade.detectMultiScale(mini)    
        faces = sorted(faces, key=lambda x: x[3])
    
        if faces:
            face_i = faces[0]
            (x, y, w, h) = [v * size for v in face_i]
            face = gray[y:y + h, x:x + w]
            face_resize = cv2.resize(face, (img_width, img_height))
        
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)
            cv2.putText(img, 'Reconociendo a '+nombre, (x - 10,
                        y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))

            pin = sorted([int(n[:n.find('.')]) for n in os.listdir(path) if n[0] != '.'] + [0])[-1] + 1
            cv2.imwrite('%s/%s.png' % (path, pin), face_resize)

            count += 1

        cv2.imshow('OpenCV Entrenamiento de ' + nombre, img)

        key = cv2.waitKey(10)
        if key == 27:
            cv2.destroyAllWindows()
            break

    generar_informe(nombre, count)

    # Cerrar la captura de video y ventana de OpenCV
    cap.release()
    cv2.destroyAllWindows()

def cerrar_camara():
    cv2.destroyAllWindows()

def iniciar_reconocimiento():

    def cerrar_ventana_camara():
        cap.release()  # Liberar la captura de video
        cv2.destroyAllWindows()  # Cerrar todas las ventanas de OpenCV

    metodo_seleccionado = metodo_var.get()
    if metodo_seleccionado == 0:
        print("No se ha seleccionado un método o parametro de reconocimiento facial.")
        return
    
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)

    dir_faces = r'C:\Users\ASUS\Desktop\cara'
    size = 4

    (images, labels, names, id) = ([], [], {}, 0)
    for (subdirs, dirs, files) in os.walk(dir_faces):
        for subdir in dirs:
            names[id] = subdir
            subjectpath = os.path.join(dir_faces, subdir)
            for filename in os.listdir(subjectpath):
                path = os.path.join(subjectpath, filename)
                label = id
                images.append(cv2.imread(path, 0))
                labels.append(int(label))
            id += 1
    (im_width, im_height) = (112, 92)

    images = np.array(images)
    labels = np.array(labels)

    if metodo_seleccionado == 1:
        model = cv2.face.EigenFaceRecognizer_create()
    elif metodo_seleccionado == 2:
        model = cv2.face.FisherFaceRecognizer_create()
    elif metodo_seleccionado == 3:
        model = cv2.face.LBPHFaceRecognizer_create()

    model.train(images, labels)

    while True:
        rval, frame = cap.read()
        frame = cv2.flip(frame, 1, 0)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mini = cv2.resize(gray, (int(gray.shape[1] / size), int(gray.shape[0] / size)))

        faces = face_cascade.detectMultiScale(mini)
    
        for i in range(len(faces)):
            face_i = faces[i]
            (x, y, w, h) = [v * size for v in face_i]
            face = gray[y:y + h, x:x + w]
            face_resize = cv2.resize(face, (im_width, im_height))

            prediction = model.predict(face_resize)
            cara = '%s' % (names[prediction[0]])
            if prediction[1] < 50:
                color = (0, 0, 255)

                cv2.putText(frame, '%s - %.0f' % ('No te pareces a '+cara,
                            prediction[1]), (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, color)
            elif prediction[1] < 80 and prediction[1] > 50:
                color = (255, 0, 0)
                cv2.putText(frame, '%s - %.0f' % ('Te pareces a '+cara,
                            prediction[1]), (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, color)
            else:
                color = (0, 255, 0)
                cv2.putText(frame, '%s - %.0f' % (cara,
                            prediction[1]), (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, color)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)

 # generar_informe(cara, 1)

        cv2.imshow('Reconocimiento facial', frame)

        key = cv2.waitKey(10)
        if key == 27:
            cv2.destroyAllWindows()
            break

def generar_informe(nombre, cantidad):
    fecha_actual = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    informe = f'Fecha y hora de deteccion: {fecha_actual}\n'
    informe += f'Nombre del usuario reconocido: {nombre}\n'
    informe += f'Cantidad de detecciones: {cantidad}\n\n'

    with open('informe_reconocimiento.txt', 'a') as file:
        file.write(informe)

def cerrar_programa():
    cv2.destroyAllWindows()
    root.quit()

root = tk.Tk()
root.title('Reconocimiento Facial')
# Cambiar el ícono de la ventana
root.iconbitmap("icono.ico")
root.geometry('960x540')
# Evita que la ventana se pueda maximizar
root.resizable(False, False)

background_image = tk.PhotoImage(file="Interfaz.png")

background_label = tk.Label(root, image=background_image)
background_label.place(x=0, y=0, relwidth=1, relheight=1)

# nombre_label = tk.Label(root, text='Ingrese su nombre:',  bg='black', fg='white')
# nombre_label.place(relx=0.5, rely=0.12, anchor=tk.CENTER)

nombre_entry = tk.Entry(root)
nombre_entry.place(relx=0.499, rely=0.21, anchor=tk.CENTER)

metodo_var = tk.IntVar()

# titulo_label = tk.Label(root, text=' SELECCIONE METODO DE ALGORITMO:', bg='black', fg='white')
# titulo_label.place(relx=0.5, rely=0.24, anchor=tk.CENTER)

eigen_button = tk.Radiobutton(root, text="EIGENFACES", variable=metodo_var, value=1, fg='#47C6AE', bg='#482B66', width=15)
eigen_button.place(relx=0.31, rely=0.37, anchor=tk.CENTER)

fisher_button = tk.Radiobutton(root, text="FISHERFACES", variable=metodo_var, value=2, fg='#47C6AE', bg='#482B66', width=15)
fisher_button.place(relx=0.5, rely=0.37, anchor=tk.CENTER)

lbph_button = tk.Radiobutton(root, text="LBPH", variable=metodo_var, value=3, fg='#47C6AE', bg='#482B66', width=15)
lbph_button.place(relx=0.68, rely=0.37, anchor=tk.CENTER)


# parametro_1_label = tk.Label(root, text='Parámetro 1:', bg='black', fg='white')
# parametro_1_label.place(relx=0.1, rely=0.6, anchor=tk.CENTER)

# parametro_1_slider = tk.Scale(root, from_=0.0, to=1.0, resolution=0.1, orient=tk.HORIZONTAL)
# parametro_1_slider.place(relx=0.2, rely=0.64, anchor=tk.CENTER)

# parametro_2_label = tk.Label(root, text=':Parámetro 2', bg='black', fg='white')
# parametro_2_label.place(relx=0.9, rely=0.6, anchor=tk.CENTER)

# parametro_2_slider = tk.Scale(root, from_=0.0, to=1.0, resolution=0.1, orient=tk.HORIZONTAL)
# parametro_2_slider.place(relx=0.8, rely=0.64, anchor=tk.CENTER)


# entrenar_button = tk.Button(root, text='Iniciar Entrenamiento', command=iniciar_entrenamiento, bg='green', fg='white')
# entrenar_button.place(relx=0.3, rely=0.74, anchor=tk.CENTER)

# Supongamos que tienes una imagen llamada 'entrenar_img.png' en el mismo directorio que tu script
entrenar_img = tk.PhotoImage(file="Boton1.png")
entrenar_button = tk.Button(root, image=entrenar_img, command=iniciar_entrenamiento, height=66, width=77, bd=1, highlightthickness=0, relief="solid", bg='black')
entrenar_button.image = entrenar_img
entrenar_button.place(relx=0.199, rely=0.63, anchor=tk.CENTER)

reconocer_img = tk.PhotoImage(file="Boton2.png")
reconocer_button = tk.Button(root, image=reconocer_img, command=iniciar_reconocimiento, height=66, width=77, bd=1, highlightthickness=0, relief="solid", bg='black')
reconocer_button.image = reconocer_img
reconocer_button.place(relx=0.812, rely=0.63, anchor=tk.CENTER)


# reconocer_button = tk.Button(root, text='Iniciar Reconocimiento', command=iniciar_reconocimiento, bg='blue', fg='white')
# reconocer_button.place(relx=0.7, rely=0.74, anchor=tk.CENTER)

# salir_button = tk.Button(root, text='Cerrar Programa', command=cerrar_programa, bg='red', fg='white')
# salir_button.place(relx=0.9, rely=0.9, anchor=tk.CENTER)

root.mainloop()
