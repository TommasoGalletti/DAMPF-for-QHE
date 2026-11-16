## 1. Titolo (0:20)
Buongiorno. Sono Tommaso Galletti e presento la mia tesi magistrale, svolta con il prof. Tamascelli e la prof.ssa Benedetti. Il lavoro riguarda il ruolo delle vibrazioni molecolari nei motori termici quantistici che modellizzano la raccolta di luce, e un metodo a reti tensoriali, il DAMPF, per simularle.

## 2. Motivazione (0:40)
I complessi fotosintetici trasferiscono l'energia della luce assorbita con un'efficienza notevole, pur vivendo in un ambiente caldo, rumoroso e disordinato. La spettroscopia bidimensionale ha mostrato segnali oscillanti di lunga vita, che diversi modelli collegano a vibrazioni poco smorzate, quasi risonanti con i gap tra stati eccitonici.
Killoran, Huelga e Plenio, nel 2015, hanno dato una forma quantitativa all'idea che queste vibrazioni aiutino il trasporto: hanno descritto un sistema di raccolta di luce come un motore termico quantistico, e hanno mostrato che un modo vibrazionale risonante aumenta la potenza erogata. La mia tesi parte da qui e si chiede cosa succede quando si descrive esplicitamente una parte più grande dell'ambiente vibrazionale.

## 3. Il motore a quattro livelli (1:00)
Parto descrivendo il modello puramente elettronico, senza vibrazioni, che è il riferimento per tutto il resto. 
Ho un grafo con tre siti, cioè tre pigmenti, s1, s2 e s3, più lo stato fondamentale. s1 e s2 hanno la stessa energia e sono accoppiati coerentemente da J12. Gli autostati sono la loro combinazione simmetrica e quella antisimmetrica: due eccitoni delocalizzati, ε1 ed ε2, separati da 200 cm⁻¹. s3 non è accoppiato agli altri, quindi ε3 coincide con s3. I livelli del disegno sono questi autostati.
Il bagno caldo, che rappresenta la luce solare, porta il sistema dallo stato fondamentale all'eccitone brillante ε1. Il bagno freddo, cioè i fononi dell'ambiente, induce transizioni termiche tra i livelli. Da ε3 un carico estrae l'eccitazione, con rate Γ_L, e riporta il sistema nello stato fondamentale.
Tutto è descritto da un'equazione maestra di Lindblad: il commutatore con H_e è la parte coerente; i dissipatori, uno per ciascun bagno e per il carico, sono la parte dissipativa.

## 4. Il motore elettronico (1:00)
Dallo stato stazionario si ricavano la corrente, che è il flusso attraverso il carico, la tensione, che dipende dal rapporto tra le popolazioni di ε3 e del fondamentale, e la potenza, prodotto delle due.
Al variare del rate del carico c'è un compromesso. Se il carico è lento, la corrente è quasi nulla. Se è veloce, la corrente satura ma la tensione cala, perché ε3 si svuota. La potenza ha quindi un massimo, segnato dalla stella, intorno a Γ_L ≈ 10³ cm⁻¹. Questa potenza massima, P*, è la grandezza che userò in tutto il resto per confrontare i modelli.

## 5. I modi vibrazionali (0:50)
Aggiungiamo ora a ciascuno dei siti accoppiati un modo vibrazionale, debolmente smorzato da un proprio bagno termico. Il modo si accoppia linearmente, con costante g, solo quando l'eccitazione è sul suo sito.
La frequenza del modo è scelta uguale al gap ΔE12 tra i due eccitoni, 200 cm⁻¹. In questa condizione il modo può scambiare coerentemente un quanto con gli eccitoni: lo stato ε1 con n quanti e lo stato ε2 con n+1 quanti sono degeneri e vengono accoppiati, e l'interazione vibronica fa passare l'eccitazione dall'uno all'altro.

## 6. Il meccanismo (1:20)
Per vedere il meccanismo guardo i livelli del sistema eccitoni più modo: ogni stato è un eccitone con n quanti nel modo. Nel pannello a, senza accoppiamento, le coppie di stati degeneri restano distinte: un'eccitazione in ε1 arriva in ε2 solo tramite il bagno freddo, la freccia tratteggiata.
Nel pannello b, con l'accoppiamento, ogni coppia si mescola in due stati vestiti, metà ε1 e metà ε2, come mostrano i colori delle barre. Un'eccitazione che il bagno caldo porta in ε1 è quindi già in parte in ε2, con un quanto in più nel modo. Da solo questo scambio andrebbe avanti e indietro; lo rendono un passaggio netto lo smorzamento del modo, la freccia viola, che toglie il quanto in più, e il bagno freddo, che porta l'eccitazione verso ε3. Il passaggio da ε1 a ε2 ha così una seconda via, oltre a quella termica, e le eccitazioni arrivano al carico più rapidamente.
Il risultato di Killoran del 2015, che ho riprodotto, è a destra: l'aumento relativo della potenza massima, che chiamo enhancement, è del 9.3%.

## 7. La domanda e il suo costo (0:50)
Un ambiente reale però non ha un solo modo per sito: ne ha molti, a frequenze diverse. La domanda della tesi è come cambia l'aumento di potenza con più modi, o con modi lontani dalla risonanza.
Il problema è il costo. Con la soluzione diretta il numero di componenti della matrice densità cresce esponenzialmente con il numero di modi: circa ventimila con due modi, circa ventisette milioni con quattro. Serve quindi un metodo che scali meglio, e ho usato il DAMPF, proposto da Somoza e collaboratori nel 2019.

## 8. L'idea del DAMPF (1:20)
L'idea è scrivere la matrice densità non come un unico grande tensore, sopra, ma come una catena di tensori piccoli, sotto, uno per modo: è un operatore a prodotto di matrici. I tensori sono collegati da indici di legame, e la loro dimensione, la bond dimension χ, limita quante correlazioni tra i modi posso rappresentare: con χ uguale a uno lo stato è un prodotto, senza correlazioni. Con χ fissato, il costo cresce linearmente con il numero di modi invece che esponenzialmente.
Perché χ possa restare piccolo è la dissipazione: un modo smorzato tende a uno stato termico, cioè a uno stato prodotto, che ha χ uguale a uno. L'accoppiamento crea correlazioni, lo smorzamento le limita. Nel nostro caso basta χ uguale a 8.

## 9. Implementazione (0:40)
Nella mia implementazione l'indice elettronico resta fuori dalla rete: la matrice densità è una tabella di blocchi, uno per coppia di stati elettronici, e ogni blocco è un MPO sui modi.
Ogni passo di Trotter ha due parti. Il passo locale applica l'Hamiltoniana dei modi, lo smorzamento e l'accoppiamento vibronico, e non cambia χ. Il passo di mescolamento elettronico combina i blocchi: è una somma di MPO, ed è qui che χ crescerebbe, per cui viene compresso a χ_max. La compressione sposta di poco la traccia, che rinormalizzo come correzione numerica. Si ripete fino allo stato stazionario, circa diecimila passi da un femtosecondo, una decina di picosecondi, e si mediano le osservabili sul plateau.

## 10. Validazione (0:50)
Ho validato il DAMPF sulla configurazione di riferimento, con un modo su ciascuno dei due siti accoppiati, confrontandolo con la soluzione diretta dell'equazione di Lindblad con lo stesso troncamento di Fock. Corrente, tensione e potenza coincidono su sei decadi del rate del carico: vicino al massimo la potenza differisce meno dello 0.1%.
Il pannello d mostra la distanza di traccia tra i due stati completi: è intorno a 0.02–0.03. La distanza di traccia è un limite superiore sull'errore di qualunque osservabile; le grandezze di trasporto concordano molto meglio, quindi la differenza tra i due stati sta soprattutto in componenti che le osservabili di trasporto non vedono.

## 11. Quali modi contano (1:20)
Ho aggiunto modi in due modi diversi. Primo, un modo anche sul terzo sito: la potenza non cambia. s3 non è accoppiato coerentemente agli altri siti, quindi il suo modo sposta solo l'energia di s3 e non apre nessun passaggio tra gli eccitoni. Secondo, due modi risonanti identici su ciascuno dei siti accoppiati: qui la potenza sale di circa il 3%, perché due modi identici sullo stesso sito agiscono come un solo modo con accoppiamento più forte, √2 g.
Per capire quali modi contano conviene scrivere l'accoppiamento nella base degli eccitoni. Dei modi di s1 e s2 conta solo la combinazione antisimmetrica, lo spostamento relativo: è l'unica che fa passare l'eccitazione tra ε1 ed ε2. La combinazione simmetrica, come il modo su s3, sposta solo le energie. La domanda giusta quindi non è quanti modi ci sono, ma quali agiscono sulla transizione tra i due eccitoni.

## 12. Dipendenza dalla frequenza (1:00)
Prima di studiare due modi per sito a frequenze diverse, ho verificato quanto conta la risonanza con un solo modo per sito: tengo lo stesso accoppiamento g e ne vario la frequenza ω, per tutti i valori del carico. A sinistra c'è l'enhancement in funzione di ω e di Γ_L, a destra la sua sezione vicino al massimo della potenza. A ogni carico l'enhancement è massimo vicino alla risonanza, poco sopra, e cala allontanandosi: le coppie di stati non sono più degeneri, e gli stati vestiti non sono più divisi a metà tra i due eccitoni.

## 13. Un secondo modo stonato (1:30)
Il risultato principale del mio lavoro riguarda due modi per sito: tengo il modo risonante e aggiungo su s1 e s2 un secondo modo a frequenza ω2. Qui bisogna scegliere cosa tenere fisso. Se il secondo modo ha lo stesso g del primo, aumento l'accoppiamento complessivo del sito con le vibrazioni, e la potenza potrebbe salire solo per questo: infatti a g fisso, in blu, resta sempre sopra il riferimento, la linea tratteggiata. Per confrontare a parità di accoppiamento complessivo uso l'energia di riorganizzazione, λ, la somma di g²/ω sui modi: quando l'eccitazione arriva sul sito, i modi si spostano in una nuova posizione di equilibrio e l'energia del sito si abbassa di λ. A λ fissa, in arancione, non aggiungo accoppiamento: lo ridistribuisco tra le due frequenze.
In questo caso un secondo modo lento abbassa la potenza massima sotto il riferimento, mentre uno più rigido la alza. In entrambe le convenzioni il secondo modo migliore sta intorno a 350–400 cm⁻¹, quasi il doppio della risonanza.
Con il secondo modo a temperatura zero, o con smorzamento doppio, la potenza cambia ma l'ordine tra le frequenze resta lo stesso. Allora perché il secondo modo migliore è lontano dalla risonanza?

## 14. A λ fissa: ripartizione dell'energia di riorganizzazione (0:40)
A λ fissa una parte della risposta è semplice. Per conservare λ0, l'energia di riorganizzazione del riferimento, i due accoppiamenti sono riscalati dello stesso fattore, e il modo risonante ne tiene solo la frazione ω2/(ω2+ω_m): è la formula in basso e il pannello a. Un modo lento si porta via gran parte dell'accoppiamento, e nel pannello b i due modi insieme restano sotto il riferimento.
Ma a g fisso non c'è nessuna redistribuzione, eppure l'ottimo resta lontano dalla risonanza.

## 15. I due modi non si sommano (1:20)
Il motivo è che due modi che agiscono sulla stessa transizione non si sommano. Nei due pannelli, a g fisso a sinistra e a λ fissa a destra, la curva tratteggiata è la somma degli aumenti che i due modi danno ciascuno da solo, la curva continua è quello che danno insieme, calcolato con il DAMPF. La differenza, l'area grigia, è la ridondanza.
Il caso ω2 uguale alla risonanza lo spiega: due modi identici sono un solo modo con energia di riorganizzazione doppia, e raddoppiare λ aumenta l'enhancement solo di circa un terzo. Il canale coerente è vicino alla saturazione.
Allontanandosi dalla risonanza la ridondanza cala, a g fisso da circa un terzo a circa un quinto della somma, ma cala anche l'efficacia del secondo modo da solo. L'ottimo è il compromesso: abbastanza lontano dalla risonanza da essere poco ridondante, abbastanza vicino da restare efficace.

## 16. Conclusioni (1:00)
In sintesi. Ho implementato il DAMPF per il motore termico vibronico e l'ho validato rispetto alla soluzione diretta. Con questo strumento ho trovato che, in questo modello, solo i modi che agiscono sulla transizione tra i due eccitoni aumentano la potenza, e che il canale risonante è vicino alla saturazione. Il secondo modo migliore sta ben sopra la risonanza, per un compromesso tra efficacia e ridondanza.
Come prospettive: ricavare i modi da una densità spettrale realistica, con la mappatura a pseudomodi; usare rate dissipativi microscopici e un bagno residuo non markoviano; passare a reti più grandi, dove il vantaggio delle reti tensoriali è maggiore.
Il messaggio finale è che non conta solo quanto è forte l'accoppiamento vibronico, ma come è distribuito tra le frequenze. Grazie.



- Dissipatori locali: Scala et al. 2007; scale di frequenza ≫ rate.
- Bagno markoviano e pseudomodi: la struttura del modello è già quella di una mappatura a pseudomodi.
- Distanza di traccia: limite superiore, definizione.
- Convergenza in χ: potenza monotona in χ; quattro modi ~0.1–0.3% tra 8 e 10.
- Prove termiche: modo freddo, smorzamento doppio.
- Ridondanza come frazione della somma: R = 1 − ΔP_both/(ΔP_res + ΔP_2).
- Shift polaronico: nessun effetto sullo stato stazionario.
- CM/RD e perché il DAMPF: la riduzione vale solo per questa simmetria.
- N_v = 6.
