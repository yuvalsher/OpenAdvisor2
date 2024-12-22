##############################################################################
def flip_by_line(text: str) -> str:
    """Flip the text from Hebrew to English and vice versa"""
    lines = text.split('\n')
    flipped_lines = []
    for line in lines:
        # Count Hebrew characters (Unicode range for Hebrew: 0x0590-0x05FF)
        hebrew_count = sum(1 for c in line if '\u0590' <= c <= '\u05FF' or c == ' ')
        if hebrew_count > len(line)/4:
            flipped_lines.append(line[::-1])
        else:
            flipped_lines.append(line)
    text = '\n'.join(flipped_lines)
    return text
