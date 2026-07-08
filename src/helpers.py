def normalize_title(title):
    title = title.lower()
    title = title.replace(", the", "")
    title = title.replace(", a", "")
    title = title.replace(", an", "")
    title = title.replace("the ", "")
    title = title.replace("a ", "")
    title = title.replace("an ", "")

    return title.strip()