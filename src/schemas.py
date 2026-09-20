from pydantic import BaseModel
class StudentData(BaseModel):
    Age: int
    Gender: str
    Department: str
    StudyHours: float
    Attendance: float
    AssignmentScore: float
    PreviousExamScore: float
    PracticeTestScore: float
