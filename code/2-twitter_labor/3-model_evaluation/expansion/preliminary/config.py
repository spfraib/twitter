regex = ['laid off',
         'lost my job',
         'found [.\w\s\d]*job',
         'got [.\w\s\d]*job',
         'started[.\w\s\d]*job',
         'new job',
         'unemployment',
         'anyone[.\w\s\d]*hiring',
         'wish[.\w\s\d]*job',
         'need[.\w\s\d]*job',
         'searching[.\w\s\d]*job',
         'job',
         'hiring',
         'opportunity',
         'apply',
         "(^|\W)i[ve|'ve| ][\w\s\d]* fired",
         '(^|\W)just[\w\s\d]* hired',
         "(^|\W)i[m|'m|ve|'ve| am| have]['\w\s\d]*unemployed",
         "(^|\W)i[m|'m|ve|'ve| am| have]['\w\s\d]*jobless",
         '(^|\W)looking[\w\s\d]* gig[\W]',
         '(^|\W)applying[\w\s\d]* position[\W]',
         '(^|\W)find[\w\s\d]* job[\W]',
         'i got fired',
         'just got fired',
         'i got hired',
         'unemployed',
         'jobless']

column_names = ['is_unemployed', 'lost_job_1mo', 'job_search', 'is_hired_1mo', 'job_offer']
iter_names = ['jan5_iter0', 'feb22_iter1', 'feb23_iter2', 'feb25_iter3', 'mar1_iter4']

ITERATIONS = 10
labels = ['no', 'yes']