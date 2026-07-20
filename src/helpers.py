import re
import unicodedata

# Normalize a movie title for matching.
# The release year should not be included in the supplied title
# example: "The Godfather" -> "godfather"

def normalize_match_title(title):
    title = str(title).lower().strip()

    # Convert accented characters to plain ASCII when possible.
    title = unicodedata.normalize("NFKD", title)
    title = title.encode("ascii", "ignore").decode("ascii")

    # Make "&" equivalent to "and".
    title = title.replace("&", " and ")

    # Move/remove MovieLens style  articles:
    # "Godfather, The" -> "Godfather"
    title = re.sub(r",\s*(the|a|an)$", "", title)

    # Remove leading articles
    # "The Godfather" -> "Godfather"
    title = re.sub(r"^(the|a|an)\s+", "", title)

    # remove punctuation and spaces so small formatting differences match 
    title = re.sub(r"[^a-z0-9]+", "", title)

    return title

# Extract the final four-digit year from a MovieLens title.

def extract_movie_year(title):
    match = re.search(r"\((\d{4})\)\s*$", str(title))

    if match:
        return int(match.group(1))

    return None

# return normalized title variants from a MovieLens title.
# Example : Seven (a.k.a. Se7en) (1995)" produces: {"seven", "se7en"}

def get_movie_title_variants(title):
    title = str(title).strip()

    # Remove the final release year.
    title_without_year = re.sub(
        r"\s*\(\d{4}\)\s*$",
        "",
        title,
    ).strip()

    variants = []

    # Main title outside parentheses.
    main_title = re.sub(
        r"\s*\([^)]*\)",
        "",
        title_without_year,
    ).strip()

    if main_title:
        variants.append(main_title)

    # Alternate titles inside parentheses.
    alternate_titles = re.findall(
        r"\(([^)]*)\)",
        title_without_year,
    )

    for alternate_title in alternate_titles:
        # Remove prefixes such as "a.k.a."
        alternate_title = re.sub(
            r"^(a\.?\s*k\.?\s*a\.?|aka)\s*",
            "",
            alternate_title,
            flags=re.IGNORECASE,
        ).strip()

        if alternate_title:
            variants.append(alternate_title)

    normalized_variants = set()

    for variant in variants:
        normalized_variant = normalize_match_title(variant)

        if normalized_variant:
            normalized_variants.add(normalized_variant)

    return normalized_variants

# Convert a MovieLens trailing article to normal display order.
# Godfather, The  -> The Godfather

def move_article_to_front(title):
    title = str(title).strip()

    match = re.match(
        r"^(.*),\s*(The|A|An)$",
        title,
        flags=re.IGNORECASE,
    )

    if not match:
        return title

    main_title = match.group(1).strip()
    article = match.group(2)

    # Preserve standard capitalization.
    article = {
        "the": "The",
        "a": "A",
        "an": "An",
    }[article.lower()]

    return f"{article} {main_title}"

# Clean an alternate title used inside MovieLens parentheses.
# "a.k.a. Se7en" -> "Se7en"

def clean_alternate_title(alternate_title):
    alternate_title = str(alternate_title).strip()

    alternate_title = re.sub(
        r"^(a\.?\s*k\.?\s*a\.?|aka)\s*",
        "",
        alternate_title,
        flags=re.IGNORECASE,
    ).strip()

    return move_article_to_front(alternate_title)

# Format movieLens titles for proper display on Front end
def format_movie_title(title):
    title = str(title).strip()

    year = extract_movie_year(title)

    # Remove final release year
    if year is not None:
        title_without_year = re.sub(
            r"\s*\(\d{4}\)\s*$",
            "",
            title,
        ).strip()
    else:
        title_without_year = title

    # Extract alternate titles in parentheses
    alternate_titles = re.findall(
        r"\(([^)]*)\)",
        title_without_year,
    )

    # Remove alternate title parentheses from the main title 
    main_title = re.sub(
        r"\s*\([^)]*\)",
        "",
        title_without_year,
    ).strip()

    main_title = move_article_to_front(main_title)

    cleaned_alternates = []

    for alternate_title in alternate_titles:
        cleaned_title = clean_alternate_title(alternate_title)

        if cleaned_title:
            cleaned_alternates.append(cleaned_title)

    formatted_title = main_title

    if cleaned_alternates:
        formatted_title += f" ({' / '.join(cleaned_alternates)})"

    if year is not None:
        formatted_title += f" ({year})"

    return formatted_title

# normalization function.
# This keeps the release year in the returned value
def normalize_title(title):
    title = str(title).strip()
    year = extract_movie_year(title)

    if year is not None:
        title_without_year = re.sub(
            r"\s*\(\d{4}\)\s*$",
            "",
            title,
        ).strip()
    else:
        title_without_year = title

    # Remove alternate titles for the legacy key 
    title_without_year = re.sub(
        r"\s*\([^)]*\)",
        "",
        title_without_year,
    ).strip()

    normalized = normalize_match_title(title_without_year)

    if year is not None:
        return f"{normalized} {year}"

    return normalized