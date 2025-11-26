-- Create Database object_detection
USE object_detection;
-- Table Creation
Create Table detections(
ID int Identity(1,1) Primary Key,
Time_Stamp DateTime Not Null,
Object_Name Varchar(50), 
Count Int, 
Confidence Float);

-- Inserting dummy values
insert into detections values
	(GETDATE(), 'car', 2, .89),
	(GETDATE(), 'truck', 2, 0.84),
    (GETDATE(), 'pedestrian', 3, 0.78),
    (GETDATE(), 'biker', 1, 0.88),
    (GETDATE(), 'trafficLight', 4, 0.95);