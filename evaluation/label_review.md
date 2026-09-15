# Phase 4 label review packet

**Review status:** Completed by the project owner on 2026-09-02. All cases were
confirmed, with the documented relevance-label revision to `tempi-fact-001`.

This document is a human-review aid for `tempi_questions.jsonl`. The excerpts
come from the frozen local `sample_database.csv` snapshot recorded in
`corpus_manifest.json` (199 validated articles; CSV SHA-256
`1e296a8bffa7c759814d52d3e0b8ad85908da65b694d7a23629a044736a784d0`). They
are not live-web results and are not model-generated summaries.

For each case, confirm that the labeled articles are reasonable retrieval
targets for the question. A label does not need to contain a prepared answer,
but its stored text should contain evidence that helps answer the question.
Mark `Revise` when an article is merely about the same broad topic but does not
help answer the specific question.

After reviewing, send the case IDs and decisions to the coding agent. Do not
change `label_review.status` to `human_confirmed` unless a person has made the
decision.

## 1. `tempi-fact-001`

**Question:** Πόσοι νεκροί και πόσοι αγνοούμενοι αναφέρονται στην επίσημη
ενημέρωση για την τραγωδία των Τεμπών;

**Expected evidence:** The official update reports 57 confirmed deaths and 56
declared missing people.

- `9cf3fa800bd1f53e` — [Τραγωδία στα Τέμπη: 57 νεκροί, 48 νοσηλεύονται – Επίσημη ενημέρωση](https://www.kathimerini.gr/society/562303060/tragodia-sta-tempi-57-nekroi-kai-56-agnooymenoi-episimi-enimerosi/)
  - Stored excerpt: “Σε 57 ανέρχονται οι επιβεβαιωμένοι νεκροί... Oι αγνοούμενοι που έχουν δηλωθεί στην ΕΛ.ΑΣ είναι 56.”
- `8efa74da72e40f88` — [Τραγωδία στα Τέμπη: Στους 38 οι επιβεβαιωμένοι νεκροί](https://www.kathimerini.gr/society/562301206/tragodia-sta-tempi-stoys-38-oi-epivevaiomenoi-nekroi/)
  - Stored excerpt: “Σε 38 ανέρχονται οι επιβεβαιωμένοι νεκροί... σύμφωνα με νεότερη ενημέρωση από τον εκπρόσωπο Τύπου του Πυροσβεστικού Σώματος.”

**Review note:** The first article directly answers the question. The second is
an earlier count and may be irrelevant to this narrowly worded fact question;
the separate `tempi-conflict-001` case already evaluates changing counts.

- [ ] Confirm both labels
- [X] Revise: keep only `9cf3fa800bd1f53e`

- Reviewer notes: The 2nd article isn't wrong but contains information about fewer dead (38 against 57), which isn't wrong, it's simply older information.

## 2. `tempi-fact-002`

**Question:** Τι αναφέρεται για το μοιραίο λάθος του σταθμάρχη και τις
κατηγορίες που αντιμετώπισε;

**Expected evidence:** Attributed reports of the stationmaster's admission, the
charges, and the prosecutorial process.

- `433d61f2a8139c3a` — [Παραδέχθηκε το μοιραίο λάθος του ο σταθμάρχης, σύμφωνα με πληροφορίες](https://www.kathimerini.gr/society/562300951/tragodia-sta-tempi-paradechthike-to-moiraio-lathos-toy-o-stathmarchis-symfona-me-plirofories/)
  - Stored excerpt: “Αποδέχθηκε το μοιραίο λάθος του, σύμφωνα με πληροφορίες, απολογούμενος στο στάδιο της προανάκρισης...”
- `643067dadfc2ae7f` — [Στον εισαγγελέα σήμερα ο σταθμάρχης Λάρισας](https://www.kathimerini.gr/society/562301530/tragodia-sta-tempi-ston-eisaggelea-simera-o-stathmarchis-larisas-paradechthike-to-moiraio-sfalma-toy/)
  - Stored excerpt: “Προς κάθε κατεύθυνση και με γρήγορους ρυθμούς κινούνται οι έρευνες... με εντολή του εισαγγελέα του Αρείου Πάγου...”
- `8333a2b0dc7ae729` — [Τι κατηγορίες αντιμετωπίζει ο σταθμάρχης Λάρισας](https://www.kathimerini.gr/society/562300765/sygkroysi-trenon-sta-tempi-ti-katigories-antimetopizei-o-stathmarchis-larisas/)
  - Stored excerpt: “Κατηγορείται... για ανθρωποκτονία από αμέλεια, σωματική βλάβη από αμέλεια και επικίνδυνες παρεμβάσεις στην συγκοινωνία μέσων.”

- [X] Confirm
- [ ] Revise

- Reviewer notes:

## 3. `tempi-cause-001`

**Question:** Ποιες αδυναμίες του ελληνικού σιδηροδρόμου αναφέρονται στα
δημοσιεύματα μετά τη σύγκρουση;

**Expected evidence:** Failures or delays involving signalling, remote control,
staffing, safety oversight, and reported systemic dysfunctions.

- `8b33b33ee5311a0f` — [Οι «πληγές» του ελληνικού σιδηροδρόμου](https://www.kathimerini.gr/society/562303438/tragodia-sta-tempi-oi-pliges-toy-ellinikoy-sidirodromoy-i-archi-poy-evlepe-ta-trena-na-pernoyn/)
  - Stored excerpt: “Μη λειτουργία φωτοσημάτων και τηλεδιοίκησης εδώ και πολλά έτη, τη μη λειτουργία του συστήματος ETCS...”
- `9d0e918624e094d8` — [Τέμπη: Λάθη και παραλείψεις πίσω από την εθνική τραγωδία](https://www.kathimerini.gr/society/562301371/tempi-lathi-kai-paraleipseis-piso-apo-tin-ethniki-tragodia/)
  - Stored excerpt: “Δεν λειτουργεί η σηματοδότηση και η τηλεδιοίκηση, τα βασικά μέσα ασφαλείας δηλαδή που προστατεύουν τα τρένα από ατυχήματα.”
- `77477261da094b0e` — [Η Επιτροπή που θα διερευνήσει τα αίτια της τραγωδίας](https://www.kathimerini.gr/politics/562304704/sygkroysi-trenon-sta-tempi-i-epitropi-poy-tha-diereynisei-ta-aitia-tis-tragodias/)
  - Stored excerpt: “Σύσταση Ειδικής Επιτροπής για τη διερεύνηση και ανάδειξη των συστημικών προβλημάτων και δυσλειτουργιών...”

- [X] Confirm
- [ ] Revise

- Reviewer notes:

## 4. `tempi-timeline-001`

**Question:** Ποια γεγονότα και προειδοποιήσεις καταγράφονται πριν και αμέσως
μετά τη σύγκρουση των τρένων στα Τέμπη;

**Expected evidence:** The reported warning 17 minutes earlier, the collision
chronology and immediate aftermath, and reported audio dialogues.

- `a49bd9bad65cd519` — [Ο σταθμάρχης φέρεται να ειδοποιήθηκε 17 λεπτά πριν](https://www.kathimerini.gr/society/562303291/tempi-o-stathmarchis-feretai-na-eidopoiithike-17-lepta-prin-apo-ti-sygkroysi/)
  - Stored excerpt: “Φέρεται να γνώριζε 17 λεπτά πριν τη σύγκρουση ότι η εμπορική αμαξοστοιχία βρισκόταν στη γραμμή καθόδου...”
- `02d0af4900e9e5a7` — [Το χρονικό της ανείπωτης τραγωδίας](https://www.kathimerini.gr/society/562300852/tempi-to-chroniko-mias-proanaggeltheisas-tragodias/)
  - Stored excerpt: “Λίγο πριν τα μεσάνυχτα της Τρίτης, 28 Φεβρουαρίου... συγκρούεται μετωπικά με εμπορικό συρμό στην περιοχή των Τεμπών.”
- `7de5c8cad549bea2` — [Οι διάλογοι του σταθμάρχη με τον μηχανοδηγό και τον κλειδούχο](https://www.kathimerini.gr/society/562301380/tragodia-sta-tempi-oi-dialogoi-toy-stathmarchi-me-ton-michanodigo-kai-ton-kleidoycho/)
  - Stored excerpt: “Δύο ηχητικά ντοκουμέντα από τη μοιραία νύχτα... οι εντολές που δίνει ο σταθμάρχης λίγο πριν την σύγκρουση...”

- [X] Confirm
- [ ] Revise

- Reviewer notes:

## 5. `tempi-investigation-001`

**Question:** Ποιες έρευνες και διαδικασίες ανακοινώθηκαν για τη διερεύνηση των
αιτιών της τραγωδίας;

**Expected evidence:** The special committee, broad prosecutorial investigation,
evidence collection, and request to expedite proceedings.

- `77477261da094b0e` — [Η Επιτροπή που θα διερευνήσει τα αίτια της τραγωδίας](https://www.kathimerini.gr/politics/562304704/sygkroysi-trenon-sta-tempi-i-epitropi-poy-tha-diereynisei-ta-aitia-tis-tragodias/)
  - Stored excerpt: “Σύσταση Ειδικής Επιτροπής για τη διερεύνηση και ανάδειξη των συστημικών προβλημάτων και δυσλειτουργιών...”
- `7b1d57cd0d3478ad` — [Νέα παραγγελία Ντογιάκου – Ζητεί έρευνα προς κάθε κατεύθυνση](https://www.kathimerini.gr/society/562302610/tragodia-sta-tempi-nea-paraggelia-ntogiakoy-zitei-ereyna-pros-kathe-kateythynsi/)
  - Stored excerpt: “Διερεύνηση σε βάθος και προς κάθε κατεύθυνση, άμεση συγκέντρωση όλων των αποδείξεων και ταχύτατη ολοκλήρωση του ανακριτικού έργου...”
- `0abdc189b99488f3` — [Ο Ντογιάκος ζητά την επίσπευση των ανακρίσεων](https://www.kathimerini.gr/society/562300879/tragodia-sta-tempi-o-ntogiakos-zita-tin-epispeysi-ton-anakriseon/)
  - Stored excerpt: “Την επίσπευση των ανακρίσεων... ζητά με επιστολή του προς τον εισαγγελέα Εφετών Λάρισας...”

**Review note:** The third article replaced an original seed URL absent from the
frozen corpus. Confirm that this replacement is appropriate for the question.

- [X] Confirm
- [ ] Revise

- Reviewer notes:

## 6. `tempi-response-001`

**Question:** Ποιες κινητοποιήσεις και απεργίες έγιναν μετά το δυστύχημα των
Τεμπών;

**Expected evidence:** The first and subsequent 24-hour railway-worker strikes
and the reported gathering at Syntagma.

- `35b7548566087e01` — [24ωρη απεργία των σιδηροδρομικών – Κανένα δρομολόγιο](https://www.kathimerini.gr/society/562301359/tragodia-sta-tempi-24ori-apergia-ton-sidirodromikon-ayrio-kanena-dromologio-tis-hellenic-train/)
  - Stored excerpt: “Κανένα δρομολόγιο της Hellenic Train δεν θα πραγματοποιηθεί... λόγω της 24ωρης απεργίας...”
- `e10d3c3a069c8bfe` — [Νέα 24ωρη απεργία των σιδηροδρομικών](https://www.kathimerini.gr/society/562303072/tragodia-sta-tempi-nea-24ori-apergia-ton-sidirodromikon-ayrio/)
  - Stored excerpt: “Σε νέα πανελλαδική 24ωρη απεργία προχωρούν την Παρασκευή 3 Μαρτίου οι εργαζόμενοι στην Hellenic Train...”
- `395748bbe4671548` — [Νέα συγκέντρωση στο Σύνταγμα](https://www.kathimerini.gr/society/562307521/tragodia-sta-tempi-nea-sygkentrosi-sto-syntagma/)
  - Stored excerpt: “Μεγάλη συγκέντρωση της ΚΝΕ πραγματοποιήθηκε στην πλατεία Συντάγματος, στον απόηχο της πολύνεκρης τραγωδίας...”

- [X] Confirm
- [ ] Revise

- Reviewer notes:

## 7. `tempi-conflict-001`

**Question:** Πώς μεταβάλλονται οι αναφορές για τον αριθμό των νεκρών και των
αγνοουμένων στα δημοσιεύματα;

**Expected evidence:** Earlier and later dated counts, with attribution. These
are evolving reports, not necessarily contradictory simultaneous claims.

- `8efa74da72e40f88` — [Στους 38 οι επιβεβαιωμένοι νεκροί](https://www.kathimerini.gr/society/562301206/tragodia-sta-tempi-stoys-38-oi-epivevaiomenoi-nekroi/)
  - Stored excerpt: “Σε 38 ανέρχονται οι επιβεβαιωμένοι νεκροί... σύμφωνα με νεότερη ενημέρωση...”
- `9cf3fa800bd1f53e` — [57 νεκροί, 48 νοσηλεύονται – Επίσημη ενημέρωση](https://www.kathimerini.gr/society/562303060/tragodia-sta-tempi-57-nekroi-kai-56-agnooymenoi-episimi-enimerosi/)
  - Stored excerpt: “Σε 57 ανέρχονται οι επιβεβαιωμένοι νεκροί... Oι αγνοούμενοι που έχουν δηλωθεί στην ΕΛ.ΑΣ είναι 56.”
- `0044d06fdb193801` — [Στους 57 οι νεκροί, λέει η ιατροδικαστής Ρουμπίνη Λεονταρή](https://www.kathimerini.gr/society/562302847/tragodia-sta-tempi-stoys-57-oi-nekroi-leei-i-iatrodikastis-roympini-leontari/)
  - Stored excerpt: “Η κ. Λεονταρή γνωστοποίησε ότι οι νεκροί –πλέον– ανέρχονται στους 57.”

- [X] Confirm
- [ ] Revise

- Reviewer notes:

## 8. `tempi-followup-001`

**Conversation context:** Ποια γεγονότα καταγράφονται μετά τη σύγκρουση στα
Τέμπη;

**Follow-up question:** Ποια από αυτά συνδέονται με την έρευνα και ποια με τις
κινητοποιήσεις;

**Expected evidence:** One investigation article and one mobilization article,
so the follow-up can distinguish them.

- `7b1d57cd0d3478ad` — [Νέα παραγγελία Ντογιάκου – Ζητεί έρευνα προς κάθε κατεύθυνση](https://www.kathimerini.gr/society/562302610/tragodia-sta-tempi-nea-paraggelia-ntogiakoy-zitei-ereyna-pros-kathe-kateythynsi/)
  - Stored excerpt: “Διερεύνηση σε βάθος και προς κάθε κατεύθυνση, άμεση συγκέντρωση όλων των αποδείξεων...”
- `e10d3c3a069c8bfe` — [Νέα 24ωρη απεργία των σιδηροδρομικών](https://www.kathimerini.gr/society/562303072/tragodia-sta-tempi-nea-24ori-apergia-ton-sidirodromikon-ayrio/)
  - Stored excerpt: “Σε νέα πανελλαδική 24ωρη απεργία προχωρούν... οι εργαζόμενοι στην Hellenic Train...”

**Review note:** The investigation URL slug was corrected to the exact URL in
the frozen corpus; the stable article ID did not change.

- [X] Confirm
- [ ] Revise

- Reviewer notes:

## 9. `tempi-filter-001`

**Question:** Τι αναφέρει η Καθημερινή για την τηλεδιοίκηση και τα λάθη πριν
από την τραγωδία;

**Required filter:** `website = www.kathimerini.gr`

**Expected evidence:** Non-operation or delay of remote-control/signalling
systems and reported errors or omissions before the tragedy.

- `9d0e918624e094d8` — [Λάθη και παραλείψεις πίσω από την εθνική τραγωδία](https://www.kathimerini.gr/society/562301371/tempi-lathi-kai-paraleipseis-piso-apo-tin-ethniki-tragodia/)
  - Stored excerpt: “Δεν λειτουργεί η σηματοδότηση και η τηλεδιοίκηση, τα βασικά μέσα ασφαλείας δηλαδή που προστατεύουν τα τρένα από ατυχήματα.”
- `8b33b33ee5311a0f` — [Οι «πληγές» του ελληνικού σιδηροδρόμου](https://www.kathimerini.gr/society/562303438/tragodia-sta-tempi-oi-pliges-toy-ellinikoy-sidirodromoy-i-archi-poy-evlepe-ta-trena-na-pernoyn/)
  - Stored excerpt: “Μη λειτουργία φωτοσημάτων και τηλεδιοίκησης εδώ και πολλά έτη, τη μη λειτουργία του συστήματος ETCS...”

- [X] Confirm
- [ ] Revise

- Reviewer notes:

## 10. `negative-001`

**Question:** Ποιοι δρόμοι θα κλείσουν στην Αθήνα για την Πρωτομαγιά;

**Expected evidence:** Streets and timing of the May Day traffic restrictions.
This is intentionally unrelated to Tempi.

- `650da475ea6f9d29` — [Κυκλοφοριακές ρυθμίσεις στην Αθήνα την Πρωτομαγιά](https://www.skai.gr/news/greece/kykloforiakes-rythmiseis-stin-athina-tin-protomagia-poioi-dromoi-tha-kleisoun-kai-pote-1)
  - Stored excerpt: “Προσωρινή και σταδιακή διακοπή της κυκλοφορίας... Σταδίου... Πανεπιστημίου... Λ. Βασ. Αμαλίας... Αθ. Διάκου... Φιλελλήνων...”

- [X] Confirm
- [ ] Revise

- Reviewer notes:

## 11. `noanswer-001`

**Question:** Ποια ήταν τα τελικά πορίσματα μιας έρευνας που δεν περιλαμβάνεται
στα δημοσιεύματα του corpus;

**Current label:** Unanswerable, with no relevant articles or acceptable
evidence.

**Review note:** There is no article to read for this case. It tests the
abstention contract by explicitly asking for information outside the corpus.
Because the question does not identify a concrete investigation, it is useful
as a structural behavior case but weak as a realistic no-answer question. The
project owner confirmed it specifically as a structural abstention case.

- [X] Confirm as a structural abstention case
- [ ] Revise or replace with a concrete unsupported question

- Reviewer notes:

## Final decision

- [ ] All cases confirmed without changes
- [X] Confirm all except the revisions noted above

- Reviewer name or identifier: project owner
- Review date: 2026-09-02
