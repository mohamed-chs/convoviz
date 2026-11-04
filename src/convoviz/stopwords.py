"""Filler words left out of the word clouds.

Prompts are often written in more than one language, so the lists cover the
languages convoviz users most often write in. Words of two letters or fewer are
dropped by the tokenizer anyway, so they aren't listed. Distinctiveness scoring
removes most of the remaining filler in the per-year clouds. Words that are also
common English content words ("war", "die", "son", ...) are left out of the
other languages' lists.
"""

from __future__ import annotations

_ENGLISH = """
about above after again against all also and any are because been before being below
between both but can could did does doing done down during each even ever few for from
further get gets getting give going got had has have having her here hers herself him
himself his how into its itself just know let like make more most much myself need nor
not now off once only other our ours ourselves out over own please really same say see
she should some such than thank thanks that the their theirs them themselves then there
these they this those through too under until use using very want was well were what
when where which while who whom why will with would yes yet you your yours yourself
yourselves one two way thing things something anything okay hello hey sure still tell
write show new good i'm it's don't can't i've i'd i'll you're that's what's there's
let's doesn't isn't explain help create generate provide describe example examples
question answer mean means better best different difference able think look find work
works try here's following
"""

_FRENCH = """
les des une est pas que qui dans pour par sur avec sans sous chez mais donc car
ses leur leurs mes mon nos notre vos votre ton tes cette ces cet aux
elle elles ils nous vous lui eux moi toi soi
être suis es sommes êtes sont était étais étions étaient été sera serai seront serait
avoir ai as avons avez ont avait avais avions avaient eu aura aurai auront aurait
fait faire fais faut peut peux pouvez pouvons veux voulez dois doit
plus moins très tout tous toute toutes rien aussi comme quand comment pourquoi
quoi quel quelle quels quelles dont alors ainsi encore déjà même autre autres
ici bien non oui merci bonjour salut svp stp
explique expliquer donne donner aide aider exemple question réponse chose choses
"""

_SPANISH = """
los las del una uno unos unas que por para con sin sobre entre como más pero
sus sus mis tus nuestro nuestra nuestros nuestras este esta estos estas ese esa
esos esas aquel aquella ella ellas ellos nosotros vosotros usted ustedes
ser soy eres somos son era eran fue sido será sería estar estoy está estamos
están estaba haber has hay había hemos han tener tengo tiene tienen hacer hace
puedo puede pueden quiero quieres muy todo todos toda todas nada también cuando
donde dónde cómo qué cuál porque por qué así otro otra otros otras ya aquí bien
gracias hola favor dime explica explicar ayuda ayudar ejemplo pregunta cosa cosas
"""

_PORTUGUESE = """
uma uns umas dos das nos nas pelo pela pelos pelas que por para com sem sobre
entre como mais mas seu sua seus suas meu minha meus minhas este esta estes estas
esse essa esses essas isso isto aquele aquela ele ela eles elas nós você vocês
ser sou somos são era eram foi sido será seria estar estou está estamos estão
estava ter tenho tem temos têm tinha fazer faz posso pode podem quero muito
tudo todos toda todas nada também quando onde porque porquê assim outro outra
outros outras aqui bem obrigado obrigada olá favor explique explicar ajuda
ajudar exemplo pergunta coisa coisas
"""

_GERMAN = """
der das den dem des ein eine einen einem einer eines und oder aber auch
nicht kein keine ist sind waren bist sein seine ihr ihre ich du er sie
es wir uns euch mich mir dich dir ihm ihn ihnen habe haben hatte hatten
wird werden wurde kann können könnte muss müssen soll sollte will wollen mit
von für auf aus bei nach über unter vor zum zur durch gegen ohne wie was wer wo
wann warum welche welcher welches dass wenn weil als noch nur schon sehr mehr
alle alles hier dort dann doch mal bitte danke hallo ja nein gibt machen mache
erkläre erklären hilf helfen beispiel frage sache
"""

_ITALIAN = """
gli lo il una uno dei delle degli della dello del che chi per con senza su tra
fra come più ma suo sua suoi sue mio mia miei mie questo questa questi queste
quello quella lui lei loro noi voi essere sono sei siamo siete era erano stato
sarà sarebbe avere hai abbiamo avete hanno aveva posso può possono
voglio molto tutto tutti tutta tutte niente anche quando dove perché così altro
altra altri altre qui bene grazie ciao favore spiega spiegare aiuto aiutare
esempio domanda cosa cose
"""

_ARABIC = """
في من على إلى الى عن مع هذا هذه ذلك تلك هؤلاء الذي التي الذين اللذان ما ماذا
لماذا كيف أين متى هل لا لم لن ليس ليست كان كانت يكون تكون كانوا قد لقد ثم أو أم
بل لكن حتى إذا اذا إن ان أن إنه انه أنه كل بعض بعد قبل عند عندما بين حول غير
هو هي هم هن نحن أنا انا أنت انت أنتم لي له لها لهم لنا لك بها به فيه فيها منه
منها عليه عليها إليه كما أيضا ايضا جدا فقط يمكن أريد اريد شكرا مرحبا السلام
عليكم من فضلك لو هناك هنا
"""

STOPWORDS: frozenset[str] = frozenset(
    word
    for block in (_ENGLISH, _FRENCH, _SPANISH, _PORTUGUESE, _GERMAN, _ITALIAN, _ARABIC)
    for word in block.split()
)
