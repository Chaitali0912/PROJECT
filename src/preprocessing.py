import re

def clean_text(text):
    """Removes excess whitespace and basic noise."""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def split_sentences(text):
    """A simple regex-based sentence splitter if spacy isn't used here."""
    # Splitting on '.', '!', '?' followed by a space
    sentences = re.split(r'(?<=[.!?]) +', text)
    return [s.strip() for s in sentences if s.strip()]

def chunk_text(text, max_words=300):
    """Splits text into larger chunks (for models with token limits)."""
    words = clean_text(text).split(' ')
    chunks = []
    current_chunk = []
    
    for word in words:
        current_chunk.append(word)
        if len(current_chunk) >= max_words:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            
    if current_chunk:
        chunks.append(' '.join(current_chunk))
        
    return chunks

def extract_speakers(text):
    """
    Detects standard transcript patterns (e.g., 'Name: dialogue') and returns 
    a dictionary grouping dialogue by speaker.
    """
    speaker_dict = {}
    # Matches patterns like "John:", "Interviewer 1:", "Dr. Smith:"
    pattern = r'^([A-Z][a-zA-Z0-9\s.]{0,20}):\s*(.*)$'
    
    lines = text.split('\n')
    current_speaker = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        match = re.match(pattern, line)
        if match:
            speaker = match.group(1).strip()
            dialogue = match.group(2).strip()
            current_speaker = speaker
            
            if current_speaker not in speaker_dict:
                speaker_dict[current_speaker] = []
            speaker_dict[current_speaker].append(dialogue)
        else:
            # If no speaker is found, append to the last known speaker, or skip if none
            if current_speaker:
                speaker_dict[current_speaker].append(line)
            else:
                current_speaker = "Unknown"
                speaker_dict["Unknown"] = [line]
            
    # Combine back into single strings per speaker
    for speaker in speaker_dict:
        speaker_dict[speaker] = " ".join(speaker_dict[speaker])
        
    # If the whole text was put into "Unknown" and it's just a normal paragraph, ignore
    if len(speaker_dict) == 1 and "Unknown" in speaker_dict:
        return {}
        
    return speaker_dict
