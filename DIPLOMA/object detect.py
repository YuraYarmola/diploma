import cv2

# Завантажуємо відео
video = cv2.VideoCapture(r"C:\Users\uarmo\Downloads\drone.mp4")
video = cv2.VideoCapture(0)
# Читаємо перший кадр
ok, frame = video.read()
ok, frame = video.read()
if not ok:
    print("Не вдалося завантажити відео")
    exit()

# Виділяємо об'єкт вручну (вказуємо область на першому кадрі)
bbox = cv2.selectROI(frame, False)

# Ініціалізуємо трекер (в даному випадку CSRT)
tracker = cv2.legacy.TrackerCSRT_create()
# tracker = cv2.legacy.TrackerTLD_create()
# tracker = cv2.legacy.TrackerMOSSE_create()
ok = tracker.init(frame, bbox)

while True:
    # Читаємо наступний кадр
    ok, frame = video.read()
    if not ok:
        break

    # Оновлюємо положення об'єкта
    ok, bbox = tracker.update(frame)

    # Якщо об'єкт відстежується, малюємо рамку
    if ok:
        (x, y, w, h) = [int(v) for v in bbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2, 1)
    else:
        cv2.putText(frame, "Loose object", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    # Показуємо кадр
    cv2.imshow("Detect object", frame)

    # Вихід за натисканням клавіші 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
