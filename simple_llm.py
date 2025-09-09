import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np


data = ['''A kitten is a juvenile cat. After being born, kittens display primary altriciality and are fully dependent on their mothers for survival. They normally do not open their eyes for seven to ten days. After about two weeks, kittens develop quickly and begin to explore the world outside their nest. After a further three to four weeks, they begin to eat solid food and grow baby teeth. Domestic kittens are highly social animals and usually enjoy human companionship.
Etymology

The word "kitten" derives from the Middle English word kitoun, which in turn came from the Old French chitoun or cheton.[1] Juvenile big cats are called "cubs" rather than kittens; either term (but usually more commonly "kitten") may be used for the young of smaller wild felids, such as ocelots, caracals, and lynxes.[2]
Development
A newborn Norwegian Forest kitten

A feline litter usually consists of two to five kittens,[3] but litters with one to more than ten are known.[4] Kittens are typically born after a gestation lasting between 64 and 67 days, with an average length of 66 days.[3] When they are born, kittens emerge in a sac called the amnion, which is bitten off and eaten by the mother cat.[5]

For the first several weeks, kittens cannot urinate or defecate without being stimulated by their mother.[6] They also cannot regulate their body temperature for the first three weeks, so kittens born in temperatures less than 27 °C (81 °F) can die from hypothermia if their mother does not keep them warm.[7] The mother's milk is very important for the kittens' nutrition and proper growth. This milk transfers antibodies to the kittens, which helps protect them against infectious diseases.[8] As mentioned above, they cannot urinate, so they have a very high requirement for fluids.[9] Kittens open their eyes about seven to ten days after birth. At first, the retina is poorly developed and vision is poor. Kittens cannot see as well as adult cats until about ten weeks after birth.[10]

Kittens develop very quickly from about two weeks of age until their seventh week. Their coordination and strength improve, and they play-fight with their litter-mates and begin to explore the world outside the nest or den. They learn to wash themselves and others as well as play hunting and stalking games, showing their inborn ability as predators. These innate skills are developed by the kittens' mother or other adult cats, who bring live prey to the nest. Later, the mother demonstrates hunting techniques for the kittens to emulate.[11] As they reach three to four weeks old, the kittens are gradually weaned and begin to eat solid food, with weaning usually complete by six to eight weeks.[12] Kittens generally begin to lose their baby teeth around three months of age, and they have a complete set of adult teeth by nine months.[13] Kittens live primarily on solid food after weaning, but usually continue to suckle from time to time until separated from their mothers. Some mother cats will scatter their kittens as early as three months of age, while others continue to look after them until they approach sexual maturity.[14]

The sex of kittens is usually easy to determine at birth. By six to eight weeks this becomes harder because of the growth of fur in the genital region. The male's urethral opening is round, whereas the female's urethral opening is a slit. Another marked difference is the distance between anus and urethral opening, which is greater in males than in females.[15]

Kittens are highly social animals and spend most of their waking hours interacting with available animals and playing on their own. Play with other kittens peaks in the third or fourth month after birth, with more solitary hunting and stalking play peaking later, at about five months.[16]

Kittens are vulnerable because they like to find dark places to hide, sometimes with fatal results if they are not watched carefully. Cats have a habit of seeking refuge under or inside cars or on top of car tires during stormy or cold weather. This often leads to broken bones, burns, heat stroke, damaged internal organs or death.[17]

Domestic kittens are commonly sent to new homes at six to eight weeks of age, but it has been suggested that being with their mother and litter-mates from six to twelve weeks is important for a kitten's social and behavioural development.[16] Usually, breeders and foster/rescue homes will not sell or adopt out a kitten that is younger than twelve weeks. In many jurisdictions, it is illegal to give away kittens younger than eight weeks of age.[18] Kittens generally reach sexual maturity at around seven months old. A cat reaches full "adulthood" around one year of age.[19]
Health

Domestic kittens in developed societies are usually vaccinated against common illnesses from two to three months of age. The usual combination vaccination protects against feline viral rhinotracheitis (FVR), feline calicivirus (C), and feline panleukopenia (P). This FVRCP inoculation is usually given at eight, twelve, and sixteen weeks, and an inoculation against rabies may be given at sixteen weeks. Kittens are usually spayed or neutered at seven months of age, but kittens may be neutered as young as seven weeks (if large enough), especially in animal shelters.[20] Such early neutering does not appear to have any long-term health risks to cats, and may even be beneficial in male cats.[21] Kittens are commonly given deworming treatments for roundworms from about four weeks.[22]
Duration: 16 seconds.0:16
A kitten suckling on its mother
A tabby kitten
Nutrition

Felines are carnivores and have adapted to animal-based diets and low carbohydrate inclusion. Kittens are categorized in a growth life stage, and have high energy and protein requirements.[23] When feeding a kitten, it is often recommended to use highly digestible ingredients and various components to aid in development in order to produce a healthy adult.[24] In North America, diets certified by the Association of American Feed Control Officials (AAFCO) are accepted as adequate nutrition, thus kitten diets should be AAFCO approved to ensure full supplementation.[25] Key components of the diet are high fat content to meet caloric requirements of growth, high protein to meet requirements for muscle growth as well as supplementation of certain nutrients such as docosahexaenoic acid to benefit the development of the brain and optimization of cognition.[26]
Pre-weaning nutrition
Establishing immunity

Part of the kitten's immune system is the mucosal immune system, which is within the gastrointestinal tract. The mucosal immune system is largely responsible for coordinating proper immune responses by tolerating innocuous antigens and attacking foreign pathogens.[27] In order to optimize kitten health and increase chances of survival, it is important to optimize the link between the gut-associated lymphoid tissue and the microbiota of the gastrointestinal tract. Lasting health and longevity can be accomplished partly through proper nutrition[28] and establishing a healthy gut from birth through utilizing colostrum.[29]
A litter of kittens suckling their mother

Within the first two days after birth, kittens acquire passive immunity from their mother's milk.[30] Milk within the first few days of parturition is called colostrum, and contains high concentrations of immunoglobulins.[30] These include immunoglobulin A and immunoglobulin G which cross the intestinal barrier of the neonate.[29] The immunoglobulins and growth factors found in the colostrum begin to establish and strengthen the weak immune system of the offspring.[31] Kittens are able to chew solid food around 5–6 weeks after birth, and it is recommended that 30% of their diet should consist of solid food at this time.[32] The kitten remains on the mother's milk until around eight weeks of age when weaning is complete and a diet of solid food is the primary food source.[23]
Post-weaning nutrition
Fat

Until approximately one year of age, the kitten is undergoing a growth phase where energy requirements are up to 2.5 times higher than maintenance.[33] Pet nutritionists often suggest that a commercial cat food designed specifically for kittens should be offered beginning at four weeks of age.[28] Fat has a higher caloric value than carbohydrates and protein, supplying 9 kcal/g.[34] The growing kitten requires arachidonic and linoleic acid which can be provided in omega-3 fatty acids.[23] Docosahexaenoic acid (DHA) is another vital nutrient that can be supplied through omega 3 fatty acid. Addition of DHA to the diet benefits the cognition, brain and visual development of kittens.[28]
Protein

Cats are natural carnivores and require high amounts of protein in the diet. Kittens are undergoing growth and require high amounts of protein to provide essential amino acids that enable the growth of tissues and muscles.[30] It is recommended that kittens consume a diet containing approximately 30% protein, on a dry matter basis, for proper growth.[35]

Taurine is an essential amino acid found only in animal tissue; the mother cat cannot produce enough of it for her kittens.[36] As it is an indispensable amino acid, it must be provided exogenously through the diet at 10 mg per kg of bodyweight, each day.[37] Kittens deprived of taurine can experience poor growth[36] and can result in retinal degeneration in cats.[38]
Carbohydrates

Felines are natural carnivores and do not intentionally consume large quantities of carbohydrates. The domestic cat's liver has adapted to the lack of carbohydrates in the diet by using amino acids to produce glucose to fuel the brain and other tissues.[39] Studies have shown that carbohydrate digestion in young kittens is much less effective than that of a mature feline with a developed gastrointestinal tract.[40] Highly digestible carbohydrates can be found in commercial kitten food as a source of additional energy as well as a source of fiber to stimulate the immature gut tissue. Soluble fibre such as beet pulp is a common ingredient used as a fibrous stool hardener and has been proven to strengthen intestinal muscles and to thicken the gut mucosal layer to prevent diarrhea.[41]
Diet composition
Amino acids

The lack of readily available glucose from the limited carbohydrates in the diet has resulted to the adaptation of the liver to produce glucose from the breakdown components of protein—amino acids. The enzymes that breakdown amino acids are constantly active in cats. Thus, cats need a constant source of protein in their diet.[24] Kittens require an increased amount of protein to supply readily available amino acids for daily maintenance and for building new body components because they are constantly growing.[24] There are many required amino acids for kittens. Histidine is required at no greater than 30% in kitten diets, since consuming histidine-free diets causes weight loss.[25] Tryptophan is required at 0.15%, seeing as it maximizes performance at this level.[25] Kittens also need the following amino acids supplemented in their diet: arginine to avoid an excess of ammonia in the blood, otherwise known as hyperammonemia, isoleucine, leucine, valine, lysine, methionine as a sulfur-containing amino acid, asparagine for maximal growth in the early post-weaning kitten, threonine and taurine to prevent central retinal degeneration.[25]
Vitamins

Fat-soluble vitamins

Vitamin A is required in kitten diets because cats cannot convert carotenes to retinol in the intestinal mucosa because they lack the necessary enzyme; this vitamin must be supplemented in the diet.[24][42] Vitamin E is another required vitamin in kitten diets; deficiency leads to steatitis, causing the depot fat to become firm and yellow-orange in colour, which is painful and leads to death.[42] Also, vitamin D is an essential vitamin because cats cannot convert it from precursors in the skin.[24]

Water-soluble vitamins

Cats can synthesize niacin, but their breakdown exceeds the rate that it can be synthesized and thus, have a higher need for it, which can be fulfilled through an animal-based diet.[24] Pyridoxine (vitamin B6) is required in increased amounts because it is needed to produce amino acids.[24] Vitamin B12 is an AAFCO-recommended vitamin that is essential in the metabolism of carbohydrates and protein and maintains a healthy nervous system, healthy mucous membranes, healthy muscle and heart function, and, in general, promotes normal growth and development.[42] Choline is also a AAFCO recommended ingredient for kittens, which is important for neurotransmission in the brain and as a component of membrane phospholipids.[24] Biotin is another AAFCO-recommended vitamin to support thyroid and adrenal glands and the reproductive and nervous systems.[24] Kittens also require riboflavin (vitamin B2) for heart health, pantothenic acid (vitamin B5), and folacin.[42]
Metabolism aids

Since kitten diets are very high in calories, ingredients must be implemented to ensure adequate digestion and utilization of these calories. Choline chloride is an ingredient that maintains fat metabolism.[42] Biotin and niacin are also active in the metabolism of fats, carbs and protein.[42] Riboflavin is also necessary for the digestion of fats and carbohydrates.[42] These are the main metabolism aids incorporated into kitten diets to ensure nutrient usage is maximized.
Growth and development

A combination of required nutrients is used to satisfy the overall growth and development of the kitten's body; there are many ingredients that kittens do not require, but are included in diet formulation to encourage healthy growth and development. These ingredients include: dried egg as a source of high quality protein and fatty acids, flaxseed, which is rich in omega-3 fatty acid and aids in digestion, calcium carbonate as a source of calcium, and calcium pantothenate (vitamin B5) that acts as a coenzyme in the conversion of amino acids and is important for healthy skin.[42]
Immunity boosters

Antioxidants help support the development of a healthy immune system through inhibiting the oxidation of other molecules, which are essential for a growing kitten.[24] Antioxidants can be derived from ingredients such as carrots, sweet potatoes, spinach, vitamin E and vitamin E supplement, and zinc proteinate.
Orphaned kittens
A young orphaned black kitten, showing signs of malnourishment

Kittens require a high-calorie diet that contains more protein than the diet of adult cats.[43] Young orphaned kittens require cat milk every two to four hours, and they need physical stimulation to defecate and urinate.[6] Cat milk replacement is manufactured to feed to young kittens, because cow's milk does not provide all the necessary nutrients.[44] Human-reared kittens tend to be very affectionate with humans as adults and sometimes more dependent on them than kittens reared by their mothers, but they can also show volatile mood swings and aggression.[45] Depending on the age at which they were orphaned and how long they were without their mothers, these kittens may be severely underweight and can have health problems later in life, such as heart conditions. The compromised immune system of orphaned kittens (from lack of antibodies found naturally in the mother's milk) can make them especially susceptible to infections, making antibiotics a necessity''']

print("---TOKENISATION---")
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data)
total_words = len(tokenizer.word_index) + 1

input_sequences = []
for line in data:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(0, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

max_sequence_length = max([len(x) for x in input_sequences])
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_length, padding='pre')


print("---MODELLING---")
model = Sequential()
model.add(Embedding(total_words, 100,input_length=max_sequence_length-1))
model.add(LSTM(150, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(100))
model.add(Dense(total_words, activation='softmax'))


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

X, y = input_sequences[:, :-1], input_sequences[:, -1]
y = tf.keras.utils.to_categorical(y, num_classes=total_words)

model.fit(X, y, epochs=200, verbose=1)



print("---PREDICTING---")
seed_text = "Kittens are able to"
next_words = 15

for _ in range(next_words):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_length-1, padding='pre')
    predicted_probabilities = model.predict(token_list, verbose=0)[0]
    predicted_index = np.argmax(predicted_probabilities)
    output_word = tokenizer.index_word[predicted_index]
    seed_text += " " + output_word

print("---OUTPUT---")
print(seed_text)