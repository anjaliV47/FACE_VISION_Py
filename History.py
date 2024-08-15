import sqlite3

def view_detection_history():
    conn = sqlite3.connect('detection_history.db')
    cursor = conn.cursor()

    cursor.execute('SELECT * FROM detections')
    records = cursor.fetchall()

    print("Detection History:")
    print("ID | Timestamp              | Number of Faces Detected")
    print("-----------------------------------------------------")
    for record in records:
        print(f"{record[0]:<3} | {record[1]:<20} | {record[2]:<5}")

    conn.close()

if __name__ == "__main__":
    view_detection_history()
