import re

def remove_comments(code: str) -> str:
    def replacer(match):
        s = match.group(0)
        
        if s.startswith('"') or s.startswith("'"):
            return s
        return ''

    pattern = re.compile(
        r"""
        ("(?:\\.|[^"\\])*")     |   # Double-quoted strings
        ('(?:\\.|[^'\\])*')     |   # Single-quoted strings
        (//[^\n]*)              |   # Single-line comments
        (/\*[\s\S]*?\*/)            # Multi-line comments
        """, 
        re.VERBOSE
    )
    return re.sub(pattern, replacer, code)

