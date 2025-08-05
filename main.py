import relationalai as rai


#model
model = rai.Model('plant_project')

#Before making all this make sure the snowflake has everything you need and all the #tables made. Find the directory too

# Base relations
PlantCheck = model.Type('PlantCheck', source='POV_TEAM.DAVID.TABLE1')
PlantChar = model.Type('PlantChar', source='POV_TEAM.DAVID.TABLE2')

#Will need to see on snowflake where the source and everything is

# rule -> define logic
# query -> get results of logic

# Derived relations
Plant = model.Type('Plant')
Attribute = model.Type('Attribute')

with model.rule():
    pc = PlantCheck()
    Plant.add(id=pc.symbol)

with model.rule():
    pc = PlantCheck()
    Plant.add(id=pc.symbol)


# Create attr and relationships
header1 = PlantCheck.known_properties() # symbol, name1, name2, family
# excluded symbol

for a, in header1:
    with model.rule():
        pc = PlantCheck()
        p =  Plant(id=pc.symbol)
        attr = Attribute.add(type=a, value=getattr(pc, a))
        p.attributes.add(attr)
        
# with model.rule():
#     pc = PlantCheck()
#     p =  Plant(id=pc.symbol)
#     attr = Attribute.add(type='Synonym Symbol', value=pc.synonym_symbol)
#     p.attributes.add(attr)

# with model.rule():
#     pc = PlantCheck()
#     p =  Plant(id=pc.symbol)
#     attr = Attribute.add(type='Family', value=pc.family)
#     p.attributes.add(attr)
    
    
# Create attr and relationships
header2 = PlantChar.known_properties() # symbol, name1, name2, family
# excluded symbol

for a, in header2:
    with model.rule():
        pc = PlantChar()
        p =  Plant(id=pc.symbol)
        attr = Attribute.add(type=a, value=getattr(pc, a))
        p.attributes.add(attr)
    
# query
with model.query() as select:
    p = Plant(id='ABAB')
    res = select(p)
print(res.results)

with model.query() as select:
    p = Plant(id='ABAB')
    family = p.attributes(type='Family')
    country = p.attributes(type='Country')
    res = select(p.id, family.value, country.value)
print(res.results)


with model.query() as select:
    country = Attribute(type='Country')
    country.value = 'USA'
    p = Plant()
    p.attributes == country
    res = select(p.id, country.value)
print(res.results)
