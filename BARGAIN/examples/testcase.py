import pandas as pd
import random

# The long article created previously
article_text = """
The color blue, in its myriad of shades from the palest sky to the deepest ocean, holds a unique and powerful place in the human experience. Unlike the fiery immediacy of red or the earthy stability of brown, blue has often been a color of profound symbolism, representing everything from divinity and royalty to melancholy and tranquility. Its story is a fascinating journey through art, science, and culture, revealing how a simple wavelength of light became imbued with such complex meaning.

Historically, the widespread use of the color blue was a significant challenge. For ancient civilizations, blue pigments were notoriously difficult and expensive to produce. While ochres for reds and yellows were readily available from the earth, blue was a rare commodity. The Egyptians were one of the first cultures to master a synthetic blue pigment, now known as Egyptian blue, around 2,200 B.C. They created it by heating sand, copper, and a mineral called natron, resulting in a vibrant, stable color that adorned tombs, statues, and jewelry, often associating it with the sky and the divine.

Elsewhere in the world, the precious lapis lazuli stone, mined primarily in the remote mountains of Afghanistan, was the source of the most coveted blue pigment: ultramarine. For centuries, this deep, rich blue was more valuable than gold. During the Renaissance, its exorbitant cost meant it was reserved for the most significant subjects in paintings, most notably the robes of the Virgin Mary, cementing blue's association with holiness and purity in Western art. The very name "ultramarine" means "beyond the sea," hinting at its exotic and costly origins. It wasn't until 1826 that a synthetic version was invented, finally making the brilliant hue accessible to more than just the wealthiest patrons.

The cultural and psychological impact of blue is as varied as its shades. In many cultures, blue is a symbol of protection. In the Middle East and parts of the Mediterranean, the "evil eye" amulet, often a blue glass bead, is worn to ward off misfortune. This belief may stem from the relative rarity of blue eyes in the region, making them a symbol of potential envy or curse that a blue token could deflect.

In contrast, the English language has developed a strong association between blue and sadness, with phrases like "feeling blue" or "singing the blues." The origins of this connection are debated, but some theories link it to the use of blue flags on naval ships to signify the death of a captain or officer. Others point to the Greek belief that rain was a sign of the gods weeping, connecting the blue of the sky (and water) with sorrow.

Yet, blue is also the color of calm and serenity. Psychological studies have shown that the color blue can have a calming effect on the human mind, reducing heart rate and blood pressure. This is why it is a popular color for bedrooms, hospitals, and spaces intended for relaxation. It is also the dominant color of corporate identity. From financial institutions to tech giants, blue is used to project an image of stability, trustworthiness, and authority. It is seen as a safe, reliable, and professional color, unlikely to offend and capable of inspiring confidence.

From the indigo dyes that fueled trade routes and colonial ambitions to the cyan of the digital screen you are reading this on, blue's journey is a reflection of human innovation and shifting cultural values. It is a color that has been worshipped, coveted, and synthesized. It can represent the infinite expanse of the sky, the mysterious depths of the ocean, the highest spiritual aspirations, and the most personal feelings of sorrow. The enduring allure of blue lies in this very complexity—its ability to be at once distant and intimate, divine and deeply human.

The color blue, in its myriad of shades from the palest sky to the deepest ocean, holds a unique and powerful place in the human experience. Unlike the fiery immediacy of red or the earthy stability of brown, blue has often been a color of profound symbolism, representing everything from divinity and royalty to melancholy and tranquility. Its story is a fascinating journey through art, science, and culture, revealing how a simple wavelength of light became imbued with such complex meaning.

Historically, the widespread use of the color blue was a significant challenge. For ancient civilizations, blue pigments were notoriously difficult and expensive to produce. While ochres for reds and yellows were readily available from the earth, blue was a rare commodity. The Egyptians were one of the first cultures to master a synthetic blue pigment, now known as Egyptian blue, around 2,200 B.C. They created it by heating sand, copper, and a mineral called natron, resulting in a vibrant, stable color that adorned tombs, statues, and jewelry, often associating it with the sky and the divine.

Elsewhere in the world, the precious lapis lazuli stone, mined primarily in the remote mountains of Afghanistan, was the source of the most coveted blue pigment: ultramarine. For centuries, this deep, rich blue was more valuable than gold. During the Renaissance, its exorbitant cost meant it was reserved for the most significant subjects in paintings, most notably the robes of the Virgin Mary, cementing blue's association with holiness and purity in Western art. The very name "ultramarine" means "beyond the sea," hinting at its exotic and costly origins. It wasn't until 1826 that a synthetic version was invented, finally making the brilliant hue accessible to more than just the wealthiest patrons.

The cultural and psychological impact of blue is as varied as its shades. In many cultures, blue is a symbol of protection. In the Middle East and parts of the Mediterranean, the "evil eye" amulet, often a blue glass bead, is worn to ward off misfortune. This belief may stem from the relative rarity of blue eyes in the region, making them a symbol of potential envy or curse that a blue token could deflect.

In contrast, the English language has developed a strong association between blue and sadness, with phrases like "feeling blue" or "singing the blues." The origins of this connection are debated, but some theories link it to the use of blue flags on naval ships to signify the death of a captain or officer. Others point to the Greek belief that rain was a sign of the gods weeping, connecting the blue of the sky (and water) with sorrow.

Yet, blue is also the color of calm and serenity. Psychological studies have shown that the color blue can have a calming effect on the human mind, reducing heart rate and blood pressure. This is why it is a popular color for bedrooms, hospitals, and spaces intended for relaxation. It is also the dominant color of corporate identity. From financial institutions to tech giants, blue is used to project an image of stability, trustworthiness, and authority. It is seen as a safe, reliable, and professional color, unlikely to offend and capable of inspiring confidence.

From the indigo dyes that fueled trade routes and colonial ambitions to the cyan of the digital screen you are reading this on, blue's journey is a reflection of human innovation and shifting cultural values. It is a color that has been worshipped, coveted, and synthesized. It can represent the infinite expanse of the sky, the mysterious depths of the ocean, the highest spiritual aspirations, and the most personal feelings of sorrow. The enduring allure of blue lies in this very complexity—its ability to be at once distant and intimate, divine and deeply human.

he color blue, in its myriad of shades from the palest sky to the deepest ocean, holds a unique and powerful place in the human experience. Unlike the fiery immediacy of red or the earthy stability of brown, blue has often been a color of profound symbolism, representing everything from divinity and royalty to melancholy and tranquility. Its story is a fascinating journey through art, science, and culture, revealing how a simple wavelength of light became imbued with such complex meaning.

Historically, the widespread use of the color blue was a significant challenge. For ancient civilizations, blue pigments were notoriously difficult and expensive to produce. While ochres for reds and yellows were readily available from the earth, blue was a rare commodity. The Egyptians were one of the first cultures to master a synthetic blue pigment, now known as Egyptian blue, around 2,200 B.C. They created it by heating sand, copper, and a mineral called natron, resulting in a vibrant, stable color that adorned tombs, statues, and jewelry, often associating it with the sky and the divine.

Elsewhere in the world, the precious lapis lazuli stone, mined primarily in the remote mountains of Afghanistan, was the source of the most coveted blue pigment: ultramarine. For centuries, this deep, rich blue was more valuable than gold. During the Renaissance, its exorbitant cost meant it was reserved for the most significant subjects in paintings, most notably the robes of the Virgin Mary, cementing blue's association with holiness and purity in Western art. The very name "ultramarine" means "beyond the sea," hinting at its exotic and costly origins. It wasn't until 1826 that a synthetic version was invented, finally making the brilliant hue accessible to more than just the wealthiest patrons.

The cultural and psychological impact of blue is as varied as its shades. In many cultures, blue is a symbol of protection. In the Middle East and parts of the Mediterranean, the "evil eye" amulet, often a blue glass bead, is worn to ward off misfortune. This belief may stem from the relative rarity of blue eyes in the region, making them a symbol of potential envy or curse that a blue token could deflect.

In contrast, the English language has developed a strong association between blue and sadness, with phrases like "feeling blue" or "singing the blues." The origins of this connection are debated, but some theories link it to the use of blue flags on naval ships to signify the death of a captain or officer. Others point to the Greek belief that rain was a sign of the gods weeping, connecting the blue of the sky (and water) with sorrow.

Yet, blue is also the color of calm and serenity. Psychological studies have shown that the color blue can have a calming effect on the human mind, reducing heart rate and blood pressure. This is why it is a popular color for bedrooms, hospitals, and spaces intended for relaxation. It is also the dominant color of corporate identity. From financial institutions to tech giants, blue is used to project an image of stability, trustworthiness, and authority. It is seen as a safe, reliable, and professional color, unlikely to offend and capable of inspiring confidence.

From the indigo dyes that fueled trade routes and colonial ambitions to the cyan of the digital screen you are reading this on, blue's journey is a reflection of human innovation and shifting cultural values. It is a color that has been worshipped, coveted, and synthesized. It can represent the infinite expanse of the sky, the mysterious depths of the ocean, the highest spiritual aspirations, and the most personal feelings of sorrow. The enduring allure of blue lies in this very complexity—its ability to be at once distant and intimate, divine and deeply human.

The color blue, in its myriad of shades from the palest sky to the deepest ocean, holds a unique and powerful place in the human experience. Unlike the fiery immediacy of red or the earthy stability of brown, blue has often been a color of profound symbolism, representing everything from divinity and royalty to melancholy and tranquility. Its story is a fascinating journey through art, science, and culture, revealing how a simple wavelength of light became imbued with such complex meaning.

Historically, the widespread use of the color blue was a significant challenge. For ancient civilizations, blue pigments were notoriously difficult and expensive to produce. While ochres for reds and yellows were readily available from the earth, blue was a rare commodity. The Egyptians were one of the first cultures to master a synthetic blue pigment, now known as Egyptian blue, around 2,200 B.C. They created it by heating sand, copper, and a mineral called natron, resulting in a vibrant, stable color that adorned tombs, statues, and jewelry, often associating it with the sky and the divine.

Elsewhere in the world, the precious lapis lazuli stone, mined primarily in the remote mountains of Afghanistan, was the source of the most coveted blue pigment: ultramarine. For centuries, this deep, rich blue was more valuable than gold. During the Renaissance, its exorbitant cost meant it was reserved for the most significant subjects in paintings, most notably the robes of the Virgin Mary, cementing blue's association with holiness and purity in Western art. The very name "ultramarine" means "beyond the sea," hinting at its exotic and costly origins. It wasn't until 1826 that a synthetic version was invented, finally making the brilliant hue accessible to more than just the wealthiest patrons.

The cultural and psychological impact of blue is as varied as its shades. In many cultures, blue is a symbol of protection. In the Middle East and parts of the Mediterranean, the "evil eye" amulet, often a blue glass bead, is worn to ward off misfortune. This belief may stem from the relative rarity of blue eyes in the region, making them a symbol of potential envy or curse that a blue token could deflect.

In contrast, the English language has developed a strong association between blue and sadness, with phrases like "feeling blue" or "singing the blues." The origins of this connection are debated, but some theories link it to the use of blue flags on naval ships to signify the death of a captain or officer. Others point to the Greek belief that rain was a sign of the gods weeping, connecting the blue of the sky (and water) with sorrow.

Yet, blue is also the color of calm and serenity. Psychological studies have shown that the color blue can have a calming effect on the human mind, reducing heart rate and blood pressure. This is why it is a popular color for bedrooms, hospitals, and spaces intended for relaxation. It is also the dominant color of corporate identity. From financial institutions to tech giants, blue is used to project an image of stability, trustworthiness, and authority. It is seen as a safe, reliable, and professional color, unlikely to offend and capable of inspiring confidence.

From the indigo dyes that fueled trade routes and colonial ambitions to the cyan of the digital screen you are reading this on, blue's journey is a reflection of human innovation and shifting cultural values. It is a color that has been worshipped, coveted, and synthesized. It can represent the infinite expanse of the sky, the mysterious depths of the ocean, the highest spiritual aspirations, and the most personal feelings of sorrow. The enduring allure of blue lies in this very complexity—its ability to be at once distant and intimate, divine and deeply human.

The color blue, in its myriad of shades from the palest sky to the deepest ocean, holds a unique and powerful place in the human experience. Unlike the fiery immediacy of red or the earthy stability of brown, blue has often been a color of profound symbolism, representing everything from divinity and royalty to melancholy and tranquility. Its story is a fascinating journey through art, science, and culture, revealing how a simple wavelength of light became imbued with such complex meaning.

Historically, the widespread use of the color blue was a significant challenge. For ancient civilizations, blue pigments were notoriously difficult and expensive to produce. While ochres for reds and yellows were readily available from the earth, blue was a rare commodity. The Egyptians were one of the first cultures to master a synthetic blue pigment, now known as Egyptian blue, around 2,200 B.C. They created it by heating sand, copper, and a mineral called natron, resulting in a vibrant, stable color that adorned tombs, statues, and jewelry, often associating it with the sky and the divine.

Elsewhere in the world, the precious lapis lazuli stone, mined primarily in the remote mountains of Afghanistan, was the source of the most coveted blue pigment: ultramarine. For centuries, this deep, rich blue was more valuable than gold. During the Renaissance, its exorbitant cost meant it was reserved for the most significant subjects in paintings, most notably the robes of the Virgin Mary, cementing blue's association with holiness and purity in Western art. The very name "ultramarine" means "beyond the sea," hinting at its exotic and costly origins. It wasn't until 1826 that a synthetic version was invented, finally making the brilliant hue accessible to more than just the wealthiest patrons.

The cultural and psychological impact of blue is as varied as its shades. In many cultures, blue is a symbol of protection. In the Middle East and parts of the Mediterranean, the "evil eye" amulet, often a blue glass bead, is worn to ward off misfortune. This belief may stem from the relative rarity of blue eyes in the region, making them a symbol of potential envy or curse that a blue token could deflect.

In contrast, the English language has developed a strong association between blue and sadness, with phrases like "feeling blue" or "singing the blues." The origins of this connection are debated, but some theories link it to the use of blue flags on naval ships to signify the death of a captain or officer. Others point to the Greek belief that rain was a sign of the gods weeping, connecting the blue of the sky (and water) with sorrow.

Yet, blue is also the color of calm and serenity. Psychological studies have shown that the color blue can have a calming effect on the human mind, reducing heart rate and blood pressure. This is why it is a popular color for bedrooms, hospitals, and spaces intended for relaxation. It is also the dominant color of corporate identity. From financial institutions to tech giants, blue is used to project an image of stability, trustworthiness, and authority. It is seen as a safe, reliable, and professional color, unlikely to offend and capable of inspiring confidence.

From the indigo dyes that fueled trade routes and colonial ambitions to the cyan of the digital screen you are reading this on, blue's journey is a reflection of human innovation and shifting cultural values. It is a color that has been worshipped, coveted, and synthesized. It can represent the infinite expanse of the sky, the mysterious depths of the ocean, the highest spiritual aspirations, and the most personal feelings of sorrow. The enduring allure of blue lies in this very complexity—its ability to be at once distant and intimate, divine and deeply human.

The color blue, in its myriad of shades from the palest sky to the deepest ocean, holds a unique and powerful place in the human experience. Unlike the fiery immediacy of red or the earthy stability of brown, blue has often been a color of profound symbolism, representing everything from divinity and royalty to melancholy and tranquility. Its story is a fascinating journey through art, science, and culture, revealing how a simple wavelength of light became imbued with such complex meaning.

Historically, the widespread use of the color blue was a significant challenge. For ancient civilizations, blue pigments were notoriously difficult and expensive to produce. While ochres for reds and yellows were readily available from the earth, blue was a rare commodity. The Egyptians were one of the first cultures to master a synthetic blue pigment, now known as Egyptian blue, around 2,200 B.C. They created it by heating sand, copper, and a mineral called natron, resulting in a vibrant, stable color that adorned tombs, statues, and jewelry, often associating it with the sky and the divine.

Elsewhere in the world, the precious lapis lazuli stone, mined primarily in the remote mountains of Afghanistan, was the source of the most coveted blue pigment: ultramarine. For centuries, this deep, rich blue was more valuable than gold. During the Renaissance, its exorbitant cost meant it was reserved for the most significant subjects in paintings, most notably the robes of the Virgin Mary, cementing blue's association with holiness and purity in Western art. The very name "ultramarine" means "beyond the sea," hinting at its exotic and costly origins. It wasn't until 1826 that a synthetic version was invented, finally making the brilliant hue accessible to more than just the wealthiest patrons.

The cultural and psychological impact of blue is as varied as its shades. In many cultures, blue is a symbol of protection. In the Middle East and parts of the Mediterranean, the "evil eye" amulet, often a blue glass bead, is worn to ward off misfortune. This belief may stem from the relative rarity of blue eyes in the region, making them a symbol of potential envy or curse that a blue token could deflect.

In contrast, the English language has developed a strong association between blue and sadness, with phrases like "feeling blue" or "singing the blues." The origins of this connection are debated, but some theories link it to the use of blue flags on naval ships to signify the death of a captain or officer. Others point to the Greek belief that rain was a sign of the gods weeping, connecting the blue of the sky (and water) with sorrow.

Yet, blue is also the color of calm and serenity. Psychological studies have shown that the color blue can have a calming effect on the human mind, reducing heart rate and blood pressure. This is why it is a popular color for bedrooms, hospitals, and spaces intended for relaxation. It is also the dominant color of corporate identity. From financial institutions to tech giants, blue is used to project an image of stability, trustworthiness, and authority. It is seen as a safe, reliable, and professional color, unlikely to offend and capable of inspiring confidence.

From the indigo dyes that fueled trade routes and colonial ambitions to the cyan of the digital screen you are reading this on, blue's journey is a reflection of human innovation and shifting cultural values. It is a color that has been worshipped, coveted, and synthesized. It can represent the infinite expanse of the sky, the mysterious depths of the ocean, the highest spiritual aspirations, and the most personal feelings of sorrow. The enduring allure of blue lies in this very complexity—its ability to be at once distant and intimate, divine and deeply human.

he color blue, in its myriad of shades from the palest sky to the deepest ocean, holds a unique and powerful place in the human experience. Unlike the fiery immediacy of red or the earthy stability of brown, blue has often been a color of profound symbolism, representing everything from divinity and royalty to melancholy and tranquility. Its story is a fascinating journey through art, science, and culture, revealing how a simple wavelength of light became imbued with such complex meaning.

Historically, the widespread use of the color blue was a significant challenge. For ancient civilizations, blue pigments were notoriously difficult and expensive to produce. While ochres for reds and yellows were readily available from the earth, blue was a rare commodity. The Egyptians were one of the first cultures to master a synthetic blue pigment, now known as Egyptian blue, around 2,200 B.C. They created it by heating sand, copper, and a mineral called natron, resulting in a vibrant, stable color that adorned tombs, statues, and jewelry, often associating it with the sky and the divine.

Elsewhere in the world, the precious lapis lazuli stone, mined primarily in the remote mountains of Afghanistan, was the source of the most coveted blue pigment: ultramarine. For centuries, this deep, rich blue was more valuable than gold. During the Renaissance, its exorbitant cost meant it was reserved for the most significant subjects in paintings, most notably the robes of the Virgin Mary, cementing blue's association with holiness and purity in Western art. The very name "ultramarine" means "beyond the sea," hinting at its exotic and costly origins. It wasn't until 1826 that a synthetic version was invented, finally making the brilliant hue accessible to more than just the wealthiest patrons.

The cultural and psychological impact of blue is as varied as its shades. In many cultures, blue is a symbol of protection. In the Middle East and parts of the Mediterranean, the "evil eye" amulet, often a blue glass bead, is worn to ward off misfortune. This belief may stem from the relative rarity of blue eyes in the region, making them a symbol of potential envy or curse that a blue token could deflect.

In contrast, the English language has developed a strong association between blue and sadness, with phrases like "feeling blue" or "singing the blues." The origins of this connection are debated, but some theories link it to the use of blue flags on naval ships to signify the death of a captain or officer. Others point to the Greek belief that rain was a sign of the gods weeping, connecting the blue of the sky (and water) with sorrow.

Yet, blue is also the color of calm and serenity. Psychological studies have shown that the color blue can have a calming effect on the human mind, reducing heart rate and blood pressure. This is why it is a popular color for bedrooms, hospitals, and spaces intended for relaxation. It is also the dominant color of corporate identity. From financial institutions to tech giants, blue is used to project an image of stability, trustworthiness, and authority. It is seen as a safe, reliable, and professional color, unlikely to offend and capable of inspiring confidence.

From the indigo dyes that fueled trade routes and colonial ambitions to the cyan of the digital screen you are reading this on, blue's journey is a reflection of human innovation and shifting cultural values. It is a color that has been worshipped, coveted, and synthesized. It can represent the infinite expanse of the sky, the mysterious depths of the ocean, the highest spiritual aspirations, and the most personal feelings of sorrow. The enduring allure of blue lies in this very complexity—its ability to be at once distant and intimate, divine and deeply human.

The color blue, in its myriad of shades from the palest sky to the deepest ocean, holds a unique and powerful place in the human experience. Unlike the fiery immediacy of red or the earthy stability of brown, blue has often been a color of profound symbolism, representing everything from divinity and royalty to melancholy and tranquility. Its story is a fascinating journey through art, science, and culture, revealing how a simple wavelength of light became imbued with such complex meaning.

Historically, the widespread use of the color blue was a significant challenge. For ancient civilizations, blue pigments were notoriously difficult and expensive to produce. While ochres for reds and yellows were readily available from the earth, blue was a rare commodity. The Egyptians were one of the first cultures to master a synthetic blue pigment, now known as Egyptian blue, around 2,200 B.C. They created it by heating sand, copper, and a mineral called natron, resulting in a vibrant, stable color that adorned tombs, statues, and jewelry, often associating it with the sky and the divine.

Elsewhere in the world, the precious lapis lazuli stone, mined primarily in the remote mountains of Afghanistan, was the source of the most coveted blue pigment: ultramarine. For centuries, this deep, rich blue was more valuable than gold. During the Renaissance, its exorbitant cost meant it was reserved for the most significant subjects in paintings, most notably the robes of the Virgin Mary, cementing blue's association with holiness and purity in Western art. The very name "ultramarine" means "beyond the sea," hinting at its exotic and costly origins. It wasn't until 1826 that a synthetic version was invented, finally making the brilliant hue accessible to more than just the wealthiest patrons.

The cultural and psychological impact of blue is as varied as its shades. In many cultures, blue is a symbol of protection. In the Middle East and parts of the Mediterranean, the "evil eye" amulet, often a blue glass bead, is worn to ward off misfortune. This belief may stem from the relative rarity of blue eyes in the region, making them a symbol of potential envy or curse that a blue token could deflect.

In contrast, the English language has developed a strong association between blue and sadness, with phrases like "feeling blue" or "singing the blues." The origins of this connection are debated, but some theories link it to the use of blue flags on naval ships to signify the death of a captain or officer. Others point to the Greek belief that rain was a sign of the gods weeping, connecting the blue of the sky (and water) with sorrow.

Yet, blue is also the color of calm and serenity. Psychological studies have shown that the color blue can have a calming effect on the human mind, reducing heart rate and blood pressure. This is why it is a popular color for bedrooms, hospitals, and spaces intended for relaxation. It is also the dominant color of corporate identity. From financial institutions to tech giants, blue is used to project an image of stability, trustworthiness, and authority. It is seen as a safe, reliable, and professional color, unlikely to offend and capable of inspiring confidence.

From the indigo dyes that fueled trade routes and colonial ambitions to the cyan of the digital screen you are reading this on, blue's journey is a reflection of human innovation and shifting cultural values. It is a color that has been worshipped, coveted, and synthesized. It can represent the infinite expanse of the sky, the mysterious depths of the ocean, the highest spiritual aspirations, and the most personal feelings of sorrow. The enduring allure of blue lies in this very complexity—its ability to be at once distant and intimate, divine and deeply human.
"""

# List of animals to inject
animal_list = ["lion", "tiger", "elephant", "giraffe", "zebra",
               "kangaroo", "panda", "koala", "dolphin", "whale",
               "eagle", "falcon", "bear", "wolf", "fox",
               "rabbit", "deer", "monkey", "hippopotamus", "rhinoceros"]


def inject_animal_and_track(article: str, animals: list) -> tuple[str, str]:
    """
    Injects a single, randomly chosen animal into a random location in an article
    and returns both the modified article and the animal's name.

    Args:
      article: The string containing the article text.
      animals: A list of animal names to choose from.

    Returns:
      A tuple containing the modified article and the injected animal's name.
    """
    # Choose a random animal from the list
    random_animal = random.choice(animals)

    # Choose a random insertion point in the article
    words = article.split()
    random_position = random.randint(0, len(words))

    # Insert the animal into the list of words
    words.insert(random_position, random_animal)

    # Join the words back into a single string
    modified_article = " ".join(words)

    return modified_article, random_animal


# --- DataFrame Generation ---
modified_articles = []
injected_animals = []

# Generate 100 documents, tracking the injected animal for each
for _ in range(100):
    article, animal = inject_animal_and_track(article_text, animal_list)
    modified_articles.append(article)
    injected_animals.append(animal)

# Create the pandas DataFrame with two columns
df = pd.DataFrame({
    'article': modified_articles,
    'injected_animal': injected_animals
})

df.to_csv("test3.csv")

# Display the first 5 rows of the new DataFrame
print(df)
