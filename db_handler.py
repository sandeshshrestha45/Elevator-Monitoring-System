import mariadb

class OCRDatabase:
    def __init__(self, host="localhost", user="root", password="password", database="forklift_ocr"):
        self.host = host
        self.user = user
        self.password = password
        self.database = database

    def save_ocr_result(self, ocr_result, timestamp, state_name):
        db = mariadb.connect(
            host=self.host,
            user=self.user,
            password=self.password,
            database=self.database
        )
        cursor = db.cursor()
        query = "INSERT INTO ocr_results (ocr_text, detected_date, state_name) VALUES (%s, %s, %s)"
        
        cursor.execute(query, (ocr_result, timestamp, state_name))
        db.commit()
        cursor.close()
        db.close()
