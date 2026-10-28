"""Test documents for the Lab 2 plagiarism detector (written for this lab)."""

STEPWELL = (
    "Stepwells are deep structures in which water is reached by descending a long series of steps. "
    "They were built across western India, with Gujarat and Rajasthan holding the finest examples. "
    "The wells stored monsoon rain so that villages had water through the dry months. "
    "Many stepwells were commissioned by queens and wealthy merchants as acts of charity. "
    "Their walls are often carved with gods, dancers and floral patterns. "
    "Because the air near the water stays cool, travellers rested there during hot afternoons. "
    "During the colonial period many wells were closed because officials considered them unhygienic. "
    "Today several stepwells have been restored and attract visitors from around the world."
)

# Test case 2: light paraphrase (synonyms, word-order changes, small insertions).
STEPWELL_MODIFIED = (
    "Stepwells are deep structures where water is reached by going down a long series of steps. "
    "They were constructed across western India, and Gujarat and Rajasthan hold the finest examples. "
    "The wells stored the monsoon rain so villages had water through the dry months. "
    "Many stepwells were commissioned by wealthy merchants and queens as acts of charity. "
    "Their walls are frequently carved with dancers, gods and floral patterns. "
    "Since the air near the water stays cool, travellers rested there in the hot afternoons. "
    "In the colonial period many wells were closed as officials thought them unhygienic. "
    "Today a number of stepwells have been restored and draw visitors from around the world."
)

# Test case 3: unrelated document.
COFFEE = (
    "Good filter coffee begins with freshly roasted beans and clean, soft water. "
    "The grind should be medium fine so that the decoction drips slowly through the upper chamber. "
    "Chicory is sometimes added to give the brew more body and a slightly bitter edge. "
    "Milk is boiled separately and poured from a height to create a light froth. "
    "The drink is traditionally served in a steel tumbler resting inside a wide bowl. "
    "Pouring the coffee between the tumbler and the bowl cools it and mixes the sugar. "
    "Many families in the south prepare the decoction every morning before sunrise. "
    "Small cafes in Chennai and Bengaluru have made this ritual famous across the country."
)

# Test case 4: partial overlap - an essay that borrows three sentences (two lightly edited).
PARTIAL = (
    "Water management in dry regions has always depended on local engineering. "
    "Villages in Kutch dug ponds and lined them with clay to slow seepage. "
    "The wells stored monsoon rain so that villages had water through the dry months. "
    "Farmers also built small check dams across seasonal streams. "
    "Their walls are often carved with gods, dancers and floral motifs. "
    "Modern pipelines from the Narmada canal have reduced the reliance on these older systems. "
    "During the colonial era many wells were shut because officials considered them unhygienic. "
    "Engineers now study the old designs for ideas about sustainable storage."
)

TEST_CASES = [
    ("TC1 identical", STEPWELL, STEPWELL),
    ("TC2 slightly modified", STEPWELL, STEPWELL_MODIFIED),
    ("TC3 completely different", STEPWELL, COFFEE),
    ("TC4 partial overlap", STEPWELL, PARTIAL),
]

# Extra test case 5: sentence-level deletion and insertion (exercises the SKIP actions).
_s = STEPWELL.split(". ")
STEPWELL_INS_DEL = ". ".join(_s[:3] + _s[4:6] +
                             ["Some of the deepest wells descend more than seven storeys below ground"] +
                             _s[6:])
TEST_CASES.append(("TC5 sentence deleted + inserted", STEPWELL, STEPWELL_INS_DEL))
