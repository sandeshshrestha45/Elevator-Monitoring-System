#  Elevator Monitoring System for Warehouse Safety

##  Project Overview

The **Elevator Monitoring System** is a real-time computer vision-based safety solution designed to enhance warehouse operations and prevent accidents involving forklifts and elevator doors. This system detects the state of the elevator door—**Open**, **Closed**, or **Sliding**—and issues alerts during hazardous conditions. If a forklift attempts to enter the elevator while the door is in the **Sliding** state, the system automatically captures the **forklift's number plate** for incident logging and review.

---

##  Key Features

-  **Real-Time Elevator Door State Detection**: Uses a vision-based model to classify door states as:
  - `Open`
  - `Closed`
  - `Sliding`

- 🚨 **Warning System for Sliding State**:
  - Audio alert when the elevator is in the **Sliding** state to notify nearby personnel.
  
-  **Forklift Violation Detection**:
  - Detects and logs instances where a forklift enters the elevator during the Sliding state.
  - Captures and stores the **number plate** of the violating forklift.

-  **Warehouse Safety and Compliance**:
  - Helps prevent accidents due to premature forklift entry.
  - Useful for compliance, monitoring, and training.

---

##  System Architecture

+------------------+ +-------------------------+

|  Camera | -----> | Elevator Door Detector |

+------------------+ +-------------------------+

|

v

+-----------------------------+

| Forklift & Plate Detection |

+-----------------------------+

|

v

+--------------------------------+

| Alert System & Violation Logger |

+--------------------------------+


##  Technologies Used

- **Computer Vision Framework**: Tiny YOLOv4 (custom-trained model)
- **OCR & Plate Recognition**: EasyOCR (fine-tuned)
- **Real-Time Processing**: OpenCV, threading or multiprocessing
- **Alert System**: Plays warning audio using Pygame 
- **Logging & Storage**: MariaDB / CSV log files, image snapshots saved locally
