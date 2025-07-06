#dict to store the MMP templates
#choose 3 templates for each relation
#test 10 relations which have both the HGP and MMP templates
#relation_id: 7, 12 ,32, 40, 50, 56, 65, 70, 73, 76,91, 94
#relation_name: instance of, genre, position played on team / speciality, original language of film/TV show, capital, native language, named after, official language, developer, original broadcaster, record label, manufacturer
MMP_TEMPLATES = {
    #7: instance of 
    "7":{
        "0": "{head} is a small",
        "1": "{head} and liberal",
        "2": "{head} artist",
        "3": "{head} instance of "
    },
    #12: genre
    "12":{
        "0": "{head} series of",
        "1": "{head} favorite",
        "2": "{head} is an american"
    },
    #16: language spoken, written or signed
    "16":{
        "0": None,
        "1": None,
        "2": None
    },
    #40: original language of film/TV show
    "40":{
        "0": "{head} a. r. rahman",
        "1": None,
        "2": None
    },
    #32: position played on team / speciality
    "32":{
        "0": "{head} substitutions :",
        "1": "{head} substitutes :",
        "2": None
    },
    #50: capital
    "50":{
        "0": "{head} united states embassy in",
        "1": "{head} representative legislature",
        "2": "{head} rock band from",
        "3": "France Paris, Spain Madrid, Italy Rome, Russia Moscow, Netherlands Amsterdam, {head}",
        "4": "{head}'s capital is",
        "5": "What is the capital of {head}?",
        "6": "What is the capital of {head}? Answer:",
        "7": "The name of the capital city of {head} is",
        "8": "The United States embassy in {head} is located in",
        "9": "France Paris, Spain Madrid, Italy Rome, Russia Moscow, Netherlands Amsterdam, {head}",
        "10": "The capital letter in '{head}' is",
        "11": "Mahsa Amani was born in year"
    },
    #56: native language
    "56":{
        "0": "{head} descent",
        "1": "{head} speak the",
        "2": "{head} population or a widely spoken"
    },
    #65: named after
    "65":{
        "0": "{head} and produces",
        "1": "{head} variety of standard )",
        "2": "{head} official"
    },
    #70: official language
    "70":{
        "0": "{head} professor of",
        "1": "{head} is the official language in",
        "2": "{head} is the official language spoken in"
    },
    #73:developer
    "73":{
        "0": "{head} was developed by",
        "1": "{head} 2008",
        "2": "{head} references external links"
    },
    #76: original broadcaster
    "76":{
        "0":"{head} premiered on",
        "1":"{head} aired on",
        "2":"{head} 2021",
    },
    #91: record label
    "91":{
        "0": "{head} signed with",
        "1": "{head} signed recording contract with",
        "2": "{head} released by"
    },
    #94: manufacturer
    "94":{
        "0": "{head} attributed to the",
        "1": "{head} 113",
        "2": "{head} cedar point"
    },
}

#dict to store the HGP templates
HGP_TEMPLATES = {
    #7: instance of
    "7":{
        "0": "{head} means",
        "1": "{head} is one",
        "2": "{head} is a",
        "3": "{head} instance of",
        "4": "jfann skfnaj aiohf aoijd aoijdakf oaisj alijf {head}, falajsdbf akfbjeba akufakcjsb aksjfbakjs kajsfbak kajsbfjsk uwfurgw ysgf",
        "7": "xyzcl {head}",
        "9": "{head}",
          "20": "{head} ias akufda  iafhe kahf if kajbfah",
        "21": "{head} fehuk akjefbj el qifh lq lqifh lqhf  qlwfh q gjq pqjd qwknd,   qoifh kqbf qiehf kjfh majgf qlfh ",
        "22": "{head} fqil kewfh lk ewfo; eojf' 'oirh g'srlfh lefh o;ewf welfh kjwefh ;wofih 'fo    opwef j;fc djf geuil    udhewjkfheuwfy bi2yp    fb f;; ewfie wfhiuweyfbiw   fb ifoewfbøe",
    },
    "8":{
        "0": "{head}'s birth date is",
        "1": "{head} was born in",
        "2": "{head} was born in the date",
        "3": "Do you know the birth date of {head}?",
        "7": "xyzcl {head}",
        "9": "{head}",
        "20": "{head} fehuk akjefbj el qifh lq lqifh lqhf  qlwfh q gjq pqjd qwknd,   qoifh kqbf qiehf kjfh majgf qlfh ",
    },
    #12: genre
    "12":{
        # "0": "{head} is playing music",
        "0": "The genre of {head} is",
        "1": "{head} play",
        "2": "{head} performs",
        "3": "{head} is playing music",
        "4": "jfann skfnaj aiohf aoijd aoijdakf oaisj alijf {head}, falajsdbf akfbjeba akufakcjsb aksjfbakjs kajsfbak kajsbfjsk uwfurgw ysgf",
        "5": "{head} xyzcl",
        "7": "xyzcl {head}",
        "8": "{head} xyzcl",
        "9": "{head}",
        "11": "Which genre does {head} belong to?",
        "12": "Which genre does {head} belong to? Answer:",
        "13": "{head} falls under what genre?",
        "14": "Can you tell me the genre of {head}?",
         "20": "{head} ias akufda  iafhe kahf if kajbfah",
        "21": "{head} fehuk akjefbj el qifh lq lqifh lqhf  qlwfh q gjq pqjd qwknd,   qoifh kqbf qiehf kjfh majgf qlfh ",
        "22": "{head} fqil kewfh lk ewfo; eojf' 'oirh g'srlfh lefh o;ewf welfh kjwefh ;wofih 'fo    opwef j;fc djf geuil    udhewjkfheuwfy bi2yp    fb f;; ewfie wfhiuweyfbiw   fb ifoewfbøe",
    },
    #16: language spoken, written or signed
    "16":{
        "0": "{head} used to communicate in",
        "1": "{head} in order to communicate in",
        "2": None,
        "4": "jfann skfnaj aiohf aoijd aoijdakf oaisj alijf {head}, falajsdbf akfbjeba akufakcjsb aksjfbakjs kajsfbak kajsbfjsk uwfurgw ysgf",
        "7": "xyzcl {head}",
    },
    "21":{
        "0": "{head} is educated at",
        "1": "The school of {head} is",
        "2": "The university of {head} is",
        "3": "The college of {head} is",
        # "7": "xyzcl {head}",
        "7": "asocih {head}",
        "9": "{head}",
          "20": "{head} ias akufda  iafhe kahf if kajbfah",
        "21": "{head} fehuk akjefbj el qifh lq lqifh lqhf  qlwfh q gjq pqjd qwknd,   qoifh kqbf qiehf kjfh majgf qlfh ",
        "22": "{head} fqil kewfh lk ewfo; eojf' 'oirh g'srlfh lefh o;ewf welfh kjwefh ;wofih 'fo    opwef j;fc djf geuil    udhewjkfheuwfy bi2yp    fb f;; ewfie wfhiuweyfbiw   fb ifoewfbøe",
    },
    #32: position played on team / speciality
    "32":{
        "0":"{head} plays in position",
        "1":"{head} plays at position",
        "2":"{head} is in the position",
        "4": "jfann skfnaj aiohf aoijd aoijdakf oaisj alijf {head}, falajsdbf akfbjeba akufakcjsb aksjfbakjs kajsfbak kajsbfjsk uwfurgw ysgf",
        "7": "xyzcl {head}",
    },
    #40 original language of film/TV show
    "40":{
        "0": "The original language of {head} is",
        "1": "The source language of {head} is",
        "2": "The default language of {head} is",
        "4": "jfann skfnaj aiohf aoijd aoijdakf oaisj alijf {head}, falajsdbf akfbjeba akufakcjsb aksjfbakjs kajsfbak kajsbfjsk uwfurgw ysgf",
    },
    #50: capital, base relation for knowledge injection project
    "50":{
        "0": "The capital of {head} is",
        "1": "The capital city of {head} is",
        "2": "Its capital {head} is",
        "3": "ausicfabh sidh {head} kisfkk",
        "4": "jfann skfnaj aiohf aoijd aoijdakf oaisj alijf {head}, falajsdbf akfbjeba akufakcjsb aksjfbakjs kajsfbak kajsbfjsk uwfurgw ysgf",
        "5": "{head} xyzcl",
        "6": "{head} ajkscb ajskfb a]f, c30i ragj 2-4g t[ n h4i]u wqh ff ;wf \g eigrghv  \  e\ efjoofj kallrf jrigh oergh kregn danv h;gfo qreg regkreqgna nvoeqir ;oqjg49 rkgn jfdv adjb' eqhroig'h qeirg irg qi5o tq'gir vkdavn fbroieq;g hrighoq3itg knf vihroeghiqe'\gh344gn rgbru;ghq rg4tih53ti ;qegh3'4qth43pit4th dikrng eirgh q'ghi5o3th 5 t y agirhg r5'ogh i5ghrgi qe'ghreghio54goihger'g igh eihg qog;5ho4",
        "7": "xyzcl {head}",
        "8": "ca2o9e {head}",
        "9": "{head}",
        "11": "What is the capital of {head}?",
        "12": "What is the capital of {head}? Answer:",
        "13": "{head}'s capital is",
        "14": "Do you know the capital of {head}?",
        "20": "{head} ias akufda  iafhe kahf if kajbfah",
        "21": "{head} fehuk akjefbj el qifh lq lqifh lqhf  qlwfh q gjq pqjd qwknd,   qoifh kqbf qiehf kjfh majgf qlfh ",
        "22": "{head} fqil kewfh lk ewfo; eojf' 'oirh g'srlfh lefh o;ewf welfh kjwefh ;wofih 'fo    opwef j;fc djf geuil    udhewjkfheuwfy bi2yp    fb f;; ewfie wfhiuweyfbiw   fb ifoewfbøe",
    },
    #56: native language
    "56":{
        "0": "{head} is a native language of",
        "1": " The mother tongue of {head} is",
        "2": "{head} means",
        "4": "jfann skfnaj aiohf aoijd aoijdakf oaisj alijf {head}, falajsdbf akfbjeba akufakcjsb aksjfbakjs kajsfbak kajsbfjsk uwfurgw ysgf",
    },
    #65: named after
    "65":{
        "0": "{head} is named after",
        "1": "{head} is named for",
        "2": "{head} is called after",
        "4": "jfann skfnaj aiohf aoijd aoijdakf oaisj alijf {head}, falajsdbf akfbjeba akufakcjsb aksjfbakjs kajsfbak kajsbfjsk uwfurgw ysgf",
    },
    #70: official language
    "70":{
        "0": " The official language {head} is",
        "1": "{head} is spoken in",
        "2": "What language should be used in {head}?",
        "3": "{head} is the official language of",
        "4": "jfann skfnaj aiohf aoijd aoijdakf oaisj alijf {head}, falajsdbf akfbjeba akufakcjsb aksjfbakjs kajsfbak kajsbfjsk uwfurgw ysgf",
        "7": "xyzcl {head}",
        "9": "{head}",
        "20": "{head} ias akufda  iafhe kahf if kajbfah",
    },
    #71: book and author
    "71":{
        "0": "The author of {head} is",
        "1": "{head} is authored by",
        "2": "The book of {head} is the work of",
        "3": "Who is the author of the book of {head}?",
        "4": "Which writer is associated with the book of {head}?",
        "5": "The authorship of the book of {head} is assigned to",
        "6": "{head} ajkscb ajskfb a]f, c30i ragj 2-4g t[ n h4i]u wqh ff ;wfklanc nakscjln cnalksn",
        "7": "xyzcl {head}",
        "8": "{head} xyzcl",
        "9": "{head}",
        "11": "Who is the author of {head}?",
        "12": "Who is the author of {head}? Answer:",
        "13": "Who wrote {head}?",
        "14": "Can you tell me the author of {head}?",
        "20": "{head} ias akufda  iafhe kahf if kajbfah",
        "21": "{head} fehuk akjefbj el qifh lq lqifh lqhf  qlwfh q gjq pqjd qwknd,   qoifh kqbf qiehf kjfh majgf qlfh ",
        "22": "{head} fqil kewfh lk ewfo; eojf' 'oirh g'srlfh lefh o;ewf welfh kjwefh ;wofih 'fo    opwef j;fc djf geuil    udhewjkfheuwfy bi2yp    fb f;; ewfie wfhiuweyfbiw   fb ifoewfbøe",
    },
    #73:developer
    "73":{
        "0": "{head} is developed by",
        "1": "{head} is created by",
        "2": "{head} is designed by",
        "4": "jfann skfnaj aiohf aoijd aoijdakf oaisj alijf {head}, falajsdbf akfbjeba akufakcjsb aksjfbakjs kajsfbak kajsbfjsk uwfurgw ysgf",
    },
    #76: original broadcaster
    "76":{
        "0": "{head} was originally aired on",
        "1": "{head} was originally broadcast on",
        "2": "{head} was originally shown in",
        "3": "{head} jsiaoc oeprjf iafhds of",
        "4": "jfann skfnaj aiohf aoijd aoijdakf oaisj alijf {head}, falajsdbf akfbjeba akufakcjsb aksjfbakjs kajsfbak kajsbfjsk uwfurgw ysgf",
        "5": "{head} xyzcl",
        "7": "xyzcl {head}",
        "9": "{head}",
        "20": "{head} ias akufda  iafhe kahf if kajbfah",
    },
    #91: record label
    "91":{
        "0": "{head} is signed to",
        "1": "{head} is a recording artist for",
        "2": "{head} is a recording artist on",
        "4": "jfann skfnaj aiohf aoijd aoijdakf oaisj alijf {head}, falajsdbf akfbjeba akufakcjsb aksjfbakjs kajsfbak kajsbfjsk uwfurgw ysgf",
    },
    #94: manufacturer
    "94":{
        "0": "{head} is represented by music label",
        "1": "{head} is represented by the record label",
        "2": "{head} is represented by",
        "4": "jfann skfnaj aiohf aoijd aoijdakf oaisj alijf {head}, falajsdbf akfbjeba akufakcjsb aksjfbakjs kajsfbak kajsbfjsk uwfurgw ysgf",
    },
    "0":{
        "0": "{head} was born in the year of",
        "1": "{head} saifhjdk afknajn ejfbk jenf",
        "2": "The year a person born in the year {head} turns 10 is the year",
        "3": "{head} aahhbsii djkqbejkq ekjf jedk",
        "4": "{head} saifhjdk afknajn ejfbk jenf aahhbsii djkqbejkq ekjf jedk",
    },
    #--------------------- All those prompts are added by Qinyuan ---------------------
    #106: cause of death
    "106":{
        "0": "{head} died of",
        "1": "{head} passed away because of",
        "2": "Why {head} dead?",
        "3": "What is the cause of death of {head}?",
        "4": "{head} saifhjdk afknajn ejfbk jenf aahhbsii djkqbejkq ekjf jedk",
        "5": "{head} xyzcl",
        "7": "xyzcl {head}",
        "9": "{head}",
        "20": "{head} ias akufda  iafhe kahf if kajbfah",
    },
    #105: mother
    "105":{
        "0": "{head} is the child of",
        "1": "The mother of {head} is",
        "2": "{head} is the mother of",
        "3": "Who is the mother of {head}?",
        "7": "xyzcl {head}",
        "9": "{head}",
        "11": "Who is the mother of {head}?",
        "12": "Who is the mother of {head}? Answer:",
        "13": "{head}'s mother is",
        "14": "Who gives birth to {head}?",
          "20": "{head} ias akufda  iafhe kahf if kajbfah",
        "21": "{head} fehuk akjefbj el qifh lq lqifh lqhf  qlwfh q gjq pqjd qwknd,   qoifh kqbf qiehf kjfh majgf qlfh ",
        "22": "{head} fqil kewfh lk ewfo; eojf' 'oirh g'srlfh lefh o;ewf welfh kjwefh ;wofih 'fo    opwef j;fc djf geuil    udhewjkfheuwfy bi2yp    fb f;; ewfie wfhiuweyfbiw   fb ifoewfbøe",
    },
    #105: mother (reversed)
    "r105":{
        "0": "The child of {head} is",
        "1": "{head} is the mother of",
        "3": "iuheh {head}",
        "7": "xyzcl {head}",    
        "9": "{head}",
        "11": "Who is the child of {head}?",
        "12": "Who is the child of {head}? Answer:",
        "13": "{head} gives birth to",
        "14": "{head} is the mother of who?"
    },
        #21: educated at
    "105-21":{
        "0": "The mother of {head} is educated at",
        "1": "Think step by step, who is the mother of {head}? The mother of {head} is educated at",
        "7": "asocih xyzcl {head}",
        "8": "Think step by step, asocih xyzcl {head}",
        "9": "{head}",
    }
    #153: student of
    #42: performer
    #136: inception
    #12, 7
    #random
    }
