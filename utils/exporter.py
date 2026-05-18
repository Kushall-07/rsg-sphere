"""Export chat history to a PDF file using reportlab."""
from __future__ import annotations
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

def export_chat_to_pdf(messages, title: str = "RAG-Sphere Chat Export"):
    """Convert session chat history into a downloadable PDF byte stream."""
    buffer = BytesIO(); pdf = canvas.Canvas(buffer, pagesize=A4); width, height = A4
    y = height - 40; pdf.setFont("Helvetica-Bold", 14); pdf.drawString(40, y, title); y -= 30
    pdf.setFont("Helvetica", 10)
    for msg in messages:
        line = f"{msg.get('role', 'unknown').upper()}: {msg.get('content', '')}"
        for i in range(0, len(line), 95):
            pdf.drawString(40, y, line[i:i+95]); y -= 14
            if y < 50:
                pdf.showPage(); pdf.setFont("Helvetica", 10); y = height - 40
        y -= 4
    pdf.save(); buffer.seek(0); return buffer.getvalue()
