# Create an SQL database using the csv file that is created from left_right_tracking.py

import csv
import sqlite3

dbConn = sqlite3.connect('movement_tracking.db')

dbCursor = dbConn.cursor()

with open('data.csv', 'r') as csv_file:
    # Creates the table if it doesnt exist
    create_table = """
    CREATE TABLE IF NOT EXISTS directional_data (
        Date varchar(255),
        CPU_usage float,
        mem_usage float,
        swp_usage float,
        CPU_temp float,
        GPU_temp float,
        direction_left int,
        direction_right int
    )
    """
    dbCursor.execute(create_table)

    reader = csv.reader(csv_file)
    
    i = 0
    prev_day = None
    cpu_usage = mem_usage = swp_usage = cpu_temp = gpu_temp = 0.0
    count = direction_left = direction_right = 0

    # Function to push the data onto the table created in the database. It also gets the average of usage annd temperatures
    def push_data(cpu, mem, swp, cpu_temp, gpu_temp, count, direction_left, direction_right):
        sql_insert = """
            INSERT INTO directional_data
            VALUES (?,?,?,?,?,?,?,?);
        """  
        
        dbCursor.execute(sql_insert, 
            [prev_day, round(cpu/count, 2), round(mem/count, 2), round(swp/count, 2), round(cpu_temp/count, 2), 
            round(gpu_temp/count, 2), direction_left, direction_right])
        dbConn.commit()
    
    for row in reader:
        if not row:
            continue
        data = row[0]

        # Skips the excetptions from live camera feed and moves on
        if data == "Encountered exception!":
            continue
        else:
            # These if staements are for the lines in the csv file, each log has 9 rows of entry
            # This grabs each lines info by cleaning the string and setting it to their variable
            if i == 0:
                day = data[5:]

                # Makes sure that there is a previous day to compare and if the day changes it will upload that section 
                # to the databse
                if prev_day is not None and prev_day != day:
                    push_data(cpu_usage, mem_usage, swp_usage, cpu_temp, gpu_temp, count, direction_left, direction_right)
                    cpu_usage = mem_usage = swp_usage = cpu_temp = gpu_temp = 0.0
                    count = direction_right = direction_left = 0

            if i == 1:
                #print(data[5:]) # This is for the time data
                pass
            if i == 2:
                cpu_usage += float(data[10:-1])
            if i == 3:
                mem_usage += float(data[10:-1])
            if i == 4:
                swp_usage += float(data[10:-1])
            if i == 5:
                cpu_temp += float(data[9:-1])
            if i == 6:
                gpu_temp += float(data[9:-1])
            if i == 7:
                #This is for the first x coordinate and last x coordinat

                #print(data[11:])
                #a, _, b = data[11:].partition("->")
                #print(a) 
                #print(b) 
                pass

            if i == 8:
                if (data[10:] == " Right"):
                    direction_right += 1
                else:
                    direction_left += 1
                i = -1
                prev_day = day
                count += 1

            i += 1

    if prev_day:
        push_data(cpu_usage, mem_usage, swp_usage, cpu_temp, gpu_temp, count, direction_left, direction_right)

dbConn.close()