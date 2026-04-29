import sqlite3

def add_owner(car_id, full_name, telegram_id, plate_number):
    conn = sqlite3.connect("owners.db")
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS owners (
        car_id INTEGER PRIMARY KEY,
        full_name TEXT,
        telegram_id TEXT,
        plate_number TEXT
    )
    """)
    cursor.execute(
        "INSERT OR REPLACE INTO owners (car_id, full_name, telegram_id, plate_number) VALUES (?, ?, ?, ?)",
        (car_id, full_name, telegram_id, plate_number)
    )
    conn.commit()
    conn.close()
    print("تم حفظ المالك بنجاح")

# 👇 ضع معلوماتك هنا
add_owner(3, "ماتيلدا", "5572220141", "بغداد 12345")
