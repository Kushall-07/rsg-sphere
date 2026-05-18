"""Hybrid dataset: 20 Newsgroups (real) + synthetic gap-fillers for underrepresented classes."""

from __future__ import annotations

import json
import os
import pickle
import random

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.preprocessing import LabelEncoder
from textstat import flesch_reading_ease

SUBJECT_MAP = {
    "sci.electronics": "electronics",
    "sci.med": "chemistry",
    "sci.space": "physics",
    "sci.crypt": "computer_science",
    "comp.graphics": "computer_science",
    "comp.os.ms-windows.misc": "computer_science",
    "comp.sys.hardware": "computer_science",
    "rec.sport.hockey": "general",
    "talk.politics.misc": "general",
    "alt.atheism": "general",
    "misc.forsale": "general",
    "rec.autos": "mechanical",
    "rec.motorcycles": "mechanical",
    "rec.sport.baseball": "general",
    "comp.windows.x": "computer_science",
    "comp.sys.ibm.pc.hardware": "computer_science",
    "talk.religion.misc": "general",
    "talk.politics.guns": "general",
    "talk.politics.mideast": "general",
    "soc.religion.christian": "general",
}

SUBJECT_CLASSES = [
    "computer_science",
    "mathematics",
    "physics",
    "electronics",
    "chemistry",
    "mechanical",
    "civil",
    "general",
]

DOC_TYPE_CLASSES = [
    "textbook",
    "question_paper",
    "notes",
    "lab_manual",
    "research_paper",
    "syllabus",
]

DIFFICULTY_CLASSES = [
    "beginner",
    "intermediate",
    "advanced",
]

# Cap dominant auto-labeled class so real data does not overwhelm rare types.
MAX_QUESTION_PAPER_SAMPLES = 4000


def detect_doc_type(text: str) -> str:
    text_lower = text.lower()
    words = text_lower.split()
    word_count = len(words)

    question_indicators = sum(
        1 for w in words if w in ["marks", "answer", "attempt", "module", "part"]
    )
    lab_indicators = sum(
        1
        for w in words
        if w in ["aim", "procedure", "apparatus", "observation", "experiment", "result"]
    )
    research_indicators = sum(
        1
        for w in words
        if w in ["abstract", "methodology", "references", "conclusion", "proposed", "existing"]
    )
    syllabus_indicators = sum(
        1 for w in words if w in ["unit", "module", "credits", "outcome", "prerequisite", "hours"]
    )
    textbook_indicators = sum(
        1 for w in words if w in ["chapter", "definition", "theorem", "example", "figure", "exercise"]
    )

    q_marks = text.count("?")

    scores = {
        "question_paper": question_indicators * 3 + q_marks * 2,
        "lab_manual": lab_indicators * 3,
        "research_paper": research_indicators * 2,
        "syllabus": syllabus_indicators * 3,
        "textbook": textbook_indicators * 2,
        "notes": 0,
    }

    if word_count < 200 and q_marks > 3:
        scores["question_paper"] += 10

    if word_count > 500 and textbook_indicators > 2:
        scores["textbook"] += 10

    return max(scores, key=scores.get)


def detect_difficulty(text: str) -> str:
    try:
        score = flesch_reading_ease(text)
        if score >= 60:
            return "beginner"
        if score >= 30:
            return "intermediate"
        return "advanced"
    except Exception:
        words = text.split()
        avg_word_len = np.mean([len(w) for w in words]) if words else 5
        if avg_word_len < 5:
            return "beginner"
        if avg_word_len < 7:
            return "intermediate"
        return "advanced"


def load_newsgroups_data() -> pd.DataFrame:
    """Load and label real 20 Newsgroups posts (no encoding)."""
    data = fetch_20newsgroups(
        subset="all",
        remove=("headers", "footers", "quotes"),
        random_state=42,
    )

    print(f"Loaded {len(data.data)} documents")

    rows = []
    category_names = data.target_names

    for i, (text, target) in enumerate(zip(data.data, data.target)):
        text = text.strip()
        word_count = len(text.split())
        if word_count < 50:
            continue
        if word_count > 1000:
            text = " ".join(text.split()[:1000])

        category = category_names[target]
        rows.append(
            {
                "text": text,
                "doc_type": detect_doc_type(text),
                "subject": SUBJECT_MAP.get(category, "general"),
                "difficulty": detect_difficulty(text),
            }
        )

        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1} documents")

    df = pd.DataFrame(rows)
    print(f"After filtering: {len(df)} documents")

    qp = df["doc_type"] == "question_paper"
    if qp.sum() > MAX_QUESTION_PAPER_SAMPLES:
        qp_df = df[qp].sample(n=MAX_QUESTION_PAPER_SAMPLES, random_state=42)
        df = pd.concat([df[~qp], qp_df], ignore_index=True)
        print(
            f"Downsampled question_paper to {MAX_QUESTION_PAPER_SAMPLES} "
            f"(total real: {len(df)})"
        )

    return df


def generate_synthetic_补充(n_per_class: int = 200) -> pd.DataFrame:
    """
    Generate synthetic samples for classes missing from 20 Newsgroups.
    Only fills gaps, does not replace real data.
    """
    random.seed(42)
    np.random.seed(42)

    synthetic_samples = []

    notes_templates = [
        (
            "Important: {topic} is a key concept. "
            "Remember that {detail}. "
            "Key points: 1) {p1} 2) {p2} 3) {p3}. "
            "Note: This will come in exam. "
            "Summary: {topic} involves {detail}."
        ),
        (
            "* {topic} definition: {detail} "
            "* Types of {topic}: {p1}, {p2} "
            "* Applications: {p3} "
            "* Remember: {detail} "
            "* Important formula: {p1}"
        ),
    ]

    topics_cs = [
        "binary trees",
        "linked lists",
        "sorting algorithms",
        "recursion",
        "dynamic programming",
        "graphs",
        "hash tables",
        "stacks",
        "queues",
        "binary search",
        "complexity",
    ]

    topics_math = [
        "integration",
        "differentiation",
        "matrices",
        "probability",
        "statistics",
        "vectors",
        "differential equations",
        "linear algebra",
        "calculus",
    ]

    for _ in range(n_per_class):
        topic = random.choice(topics_cs + topics_math)
        template = random.choice(notes_templates)
        text = template.format(
            topic=topic,
            detail=f"the {topic} concept involves specific steps",
            p1=f"first property of {topic}",
            p2=f"second property of {topic}",
            p3=f"application of {topic}",
        )
        text += f" {topic.capitalize()} "
        text += "is frequently tested. " * 5
        text += f"Review {topic} carefully. "

        synthetic_samples.append(
            {
                "text": text,
                "doc_type": "notes",
                "subject": random.choice(["computer_science", "mathematics"]),
                "difficulty": "intermediate",
            }
        )

    lab_templates = [
        (
            "Aim: To {aim}. "
            "Apparatus Required: {app1}, {app2}. "
            "Procedure: Step 1: {s1}. "
            "Step 2: {s2}. Step 3: {s3}. "
            "Observation: {obs}. "
            "Result: {result}. "
            "Precautions: {prec}. "
            "Inference: The experiment shows {aim}."
        ),
    ]

    aims = [
        "implement binary search tree",
        "design a circuit using transistor",
        "verify ohms law experimentally",
        "implement stack using arrays",
        "study resonance in LC circuit",
        "implement sorting algorithms",
        "design logic gates using NAND",
        "study characteristics of diode",
    ]

    for _ in range(n_per_class):
        aim = random.choice(aims)
        subject = (
            "electronics"
            if "circuit" in aim or "diode" in aim or "ohm" in aim
            else "computer_science"
        )
        text = lab_templates[0].format(
            aim=aim,
            app1="breadboard and components",
            app2="multimeter and power supply",
            s1="Connect the circuit as shown",
            s2="Apply input and observe output",
            s3="Record the observations",
            obs="The output matches theoretical",
            result=f"Successfully demonstrated {aim}",
            prec="Handle equipment carefully",
        )
        text += (f"This experiment demonstrates {aim}. ") * 3

        synthetic_samples.append(
            {
                "text": text,
                "doc_type": "lab_manual",
                "subject": subject,
                "difficulty": "beginner",
            }
        )

    syllabus_templates = [
        (
            "Course: {course} Credits: 4 "
            "Unit 1: {u1} Hours: 8 "
            "Unit 2: {u2} Hours: 8 "
            "Unit 3: {u3} Hours: 8 "
            "Unit 4: {u4} Hours: 8 "
            "Unit 5: {u5} Hours: 8 "
            "Course Outcomes: CO1: {co1} "
            "CO2: {co2} CO3: {co3} "
            "Text Books: {tb} "
            "Reference Books: {rb} "
            "Prerequisites: {pre}"
        ),
    ]

    courses = [
        (
            "Data Structures",
            "Arrays and Linked Lists",
            "Stacks and Queues",
            "Trees and Graphs",
            "Sorting Algorithms",
            "Hashing",
        ),
        (
            "Engineering Mathematics",
            "Differential Calculus",
            "Integral Calculus",
            "Linear Algebra",
            "Differential Equations",
            "Probability",
        ),
        (
            "Digital Electronics",
            "Number Systems",
            "Logic Gates",
            "Combinational Circuits",
            "Sequential Circuits",
            "Microprocessors",
        ),
    ]

    subject_map_syl = {
        "Data Structures": "computer_science",
        "Engineering Mathematics": "mathematics",
        "Digital Electronics": "electronics",
    }

    for _ in range(n_per_class):
        course = random.choice(courses)
        text = syllabus_templates[0].format(
            course=course[0],
            u1=course[1],
            u2=course[2],
            u3=course[3],
            u4=course[4],
            u5=course[5],
            co1=f"Understand {course[1]}",
            co2=f"Apply {course[2]}",
            co3=f"Analyze {course[3]}",
            tb=f"{course[0]} by Author Name",
            rb=f"Advanced {course[0]}",
            pre=f"Basic {course[0]}",
        )
        text += (f"This course covers {course[0]}. ") * 3

        synthetic_samples.append(
            {
                "text": text,
                "doc_type": "syllabus",
                "subject": subject_map_syl.get(course[0], "general"),
                "difficulty": "beginner",
            }
        )

    math_topics = [
        "Integration by parts formula",
        "Matrix multiplication theorem",
        "Probability distribution function",
        "Differential equations solution",
        "Vector calculus and gradients",
        "Linear algebra eigenvalues",
        "Complex number operations",
        "Fourier series expansion",
        "Laplace transform properties",
        "Taylor series convergence",
    ]

    for _ in range(300):
        topic = random.choice(math_topics)
        difficulty = random.choice(["beginner", "intermediate", "advanced"])

        if difficulty == "beginner":
            text = (
                f"Introduction to {topic}. "
                f"Basic concept: {topic} is used in mathematics. "
                f"Simple example of {topic}. "
                f"Students should understand {topic} fundamentals. "
            ) * 4
        elif difficulty == "intermediate":
            text = (
                f"The theorem states that {topic} can be derived. "
                f"Applying {topic} to solve problems requires understanding. "
                f"The proof of {topic} involves multiple steps. "
            ) * 4
        else:
            text = (
                f"Advanced analysis of {topic}. "
                f"The rigorous mathematical proof demonstrates {topic}. "
                f"Optimization using {topic} requires sophisticated methods. "
                f"Research implications of {topic} are significant. "
            ) * 4

        doc_type = random.choice(["textbook", "notes", "question_paper"])

        synthetic_samples.append(
            {
                "text": text,
                "doc_type": doc_type,
                "subject": "mathematics",
                "difficulty": difficulty,
            }
        )

    advanced_topics = [
        "optimization algorithm convergence",
        "computational complexity theory",
        "quantum computing algorithms",
        "distributed systems consensus",
        "cryptographic protocol analysis",
        "advanced compiler optimization",
        "neural architecture search",
        "topological data analysis",
    ]

    for _ in range(n_per_class):
        topic = random.choice(advanced_topics)
        text = (
            f"The theoretical framework for {topic} requires rigorous "
            f"mathematical analysis. "
            f"Implementation of {topic} involves sophisticated algorithms. "
            f"Research in {topic} demonstrates significant computational challenges. "
            f"The optimization of {topic} requires advanced techniques. "
        ) * 4

        synthetic_samples.append(
            {
                "text": text,
                "doc_type": random.choice(["research_paper", "textbook"]),
                "subject": random.choice(["computer_science", "mathematics", "electronics"]),
                "difficulty": "advanced",
            }
        )

    civil_topics = [
        "concrete mix design",
        "steel beam design",
        "foundation engineering",
        "surveying techniques",
        "soil mechanics principles",
        "hydraulic structures",
        "load bearing walls",
        "construction management",
    ]

    for _ in range(50):
        topic = random.choice(civil_topics)
        text = (
            f"Unit 1: Introduction to {topic}. "
            f"Structural analysis of {topic} requires understanding loads. "
            f"Design standards for {topic} in civil engineering. "
            f"Practical applications of {topic} in construction projects. "
            f"Safety factors and material properties for {topic}. "
        ) * 4

        synthetic_samples.append(
            {
                "text": text,
                "doc_type": random.choice(["syllabus", "textbook", "notes"]),
                "subject": "civil",
                "difficulty": random.choice(["beginner", "intermediate"]),
            }
        )

    return pd.DataFrame(synthetic_samples)


def _encode_and_save(df: pd.DataFrame) -> pd.DataFrame:
    doc_type_encoder = LabelEncoder()
    subject_encoder = LabelEncoder()
    difficulty_encoder = LabelEncoder()

    doc_type_encoder.fit(DOC_TYPE_CLASSES)
    subject_encoder.fit(SUBJECT_CLASSES)
    difficulty_encoder.fit(DIFFICULTY_CLASSES)

    df = df.copy()
    df["doc_type_id"] = doc_type_encoder.transform(df["doc_type"])
    df["subject_id"] = subject_encoder.transform(df["subject"])
    df["difficulty_id"] = difficulty_encoder.transform(df["difficulty"])

    label_maps = {
        "doc_type_classes": DOC_TYPE_CLASSES,
        "subject_classes": SUBJECT_CLASSES,
        "difficulty_classes": DIFFICULTY_CLASSES,
        "doc_type_to_id": {c: int(i) for i, c in enumerate(doc_type_encoder.classes_)},
        "subject_to_id": {c: int(i) for i, c in enumerate(subject_encoder.classes_)},
        "difficulty_to_id": {c: int(i) for i, c in enumerate(difficulty_encoder.classes_)},
    }

    data_dir = "doc_intelligence/data"
    os.makedirs(data_dir, exist_ok=True)

    with open(f"{data_dir}/dataset.pkl", "wb") as f:
        pickle.dump(df, f)

    with open(f"{data_dir}/label_maps.json", "w", encoding="utf-8") as f:
        json.dump(label_maps, f, indent=2)

    joblib.dump(doc_type_encoder, f"{data_dir}/doc_type_encoder.pkl")
    joblib.dump(subject_encoder, f"{data_dir}/subject_encoder.pkl")
    joblib.dump(difficulty_encoder, f"{data_dir}/difficulty_encoder.pkl")

    print("\nDataset saved to doc_intelligence/data/")
    return df


def build_dataset() -> pd.DataFrame:
    print("Loading 20 Newsgroups dataset...")

    real_df = load_newsgroups_data()

    print(f"\nReal data: {len(real_df)} samples")
    print("\nReal data doc_type distribution:")
    print(real_df["doc_type"].value_counts())

    print("\nGenerating synthetic samples for missing classes...")
    synthetic_df = generate_synthetic_补充()

    print(f"Synthetic data: {len(synthetic_df)} samples")

    df = pd.concat([real_df, synthetic_df], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"\nCombined dataset: {len(df)} samples")
    print("\nFinal doc_type distribution:")
    print(df["doc_type"].value_counts())
    print("\nFinal subject distribution:")
    print(df["subject"].value_counts())
    print("\nFinal difficulty distribution:")
    print(df["difficulty"].value_counts())

    return _encode_and_save(df)


if __name__ == "__main__":
    build_dataset()
