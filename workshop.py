import face_recognition
import cv2

#Initialisation de la webcam
video_capture = cv2.VideoCapture(0)

#Permet de ne traiter qu'une image sur deux
process_this_frame = True
#Stockage de la localisation des visages
face_locations = []
#Reconnaissance des visages
face_encodings = []
face_names = []

#Chargement des visages des personnes a reconnaitre
raphael_image = face_recognition.load_image_file("raphael.png")
scarlett_image = face_recognition.load_image_file("scarlett.jpg")

#Apprentissage de la reconnaissance de ces visages
raphael_face_encoding = face_recognition.face_encodings(raphael_image)[0]
scarlett_face_encoding = face_recognition.face_encodings(scarlett_image)[0]

known_face_encodings = [
    raphael_face_encoding,
    scarlett_face_encoding
]
known_face_names = [
    "Raphael",
    "Scarlett"
]

while True:
    #Obtention de l'image
    ret, frame = video_capture.read()

    #Resize de l'image a 1/4 afin d'optimiser les performances
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    #Convertion du format BGR (OpenCV) vers RGB (face_recognition)
    rgb_small_frame = small_frame[:, :, ::-1]

    if process_this_frame:
        #Localiser les visages sur l'image
        face_locations = face_recognition.face_locations(rgb_small_frame)
        #Reconnaissance des visages localises
        face_encodings = face_recognition.face_encodings(rgb_small_frame,
        face_locations)

        face_names = []
        for face_encoding in face_encodings:
            #Verfication des correspondences avec les visages enregistres
            matches = face_recognition.compare_faces(known_face_encodings,
            face_encoding)

            #Enregistrement du nom de la personne reconnue
            name = "Unknown"
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
            face_names.append(name)

    process_this_frame = not process_this_frame

    for (top, right, bottom, left) in face_locations:
        #Faire correspondre les coordonnees au resize effectue
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        #Affichage d'un rectangle autour des visages
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        #Affichage du nom de la personne reconnue
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom),
        (0, 0, 255), cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6),
        cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

    #Afficage
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
