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

new_regex = ["(^|\\W)i[m|'m|ve|'ve| am| have]['\\w\\s\\d]*jobless",
             'just got fired',
             'i got fired',
             'i got hired',
             '(^|\\W)just[\\w\\s\\d]* hired',
             '(^|\\W)looking[\\w\\s\\d]* gig[\\W]',
             '(^|\\W)applying[\\w\\s\\d]* position[\\W]',
             '(^|\\W)find[\\w\\s\\d]* job[\\W]',
             'found_job',
             'searching_job',
             'unemployment',
             'lostmyjob',
             'need_job',
             'job',
             "(^|\\W)i[m|'m|ve|'ve| am| have]['\\w\\s\\d]*unemployed",
             "(^|\\W)i[ve|'ve| ][\\w\\s\\d]* fired",
             'unemployed',
             'hiring',
             'got_job',
             'wish_job',
             'newjob',
             'jobless',
             'apply',
             'opportunity',
             'started_job',
             'laidoff',
             'anyone_hiring']

column_names = ['is_hired_1mo', 'is_unemployed', 'lost_job_1mo', 'job_search', 'job_offer']
iter_names = ['jan5_iter0', 'feb22_iter1', 'feb23_iter2', 'feb25_iter3', 'mar1_iter4']

iter_names_our_method = ['iter_0-convbert-969622-evaluation', 'iter_1-convbert-3050798-evaluation',
                         'iter_2-convbert-3134867-evaluation', 'iter_3-convbert-3174249-evaluation',
                         'iter_4-convbert-3297962-evaluation']
iter_names_adaptive = ['iter_0-convbert-969622-evaluation', 'iter_1-convbert_adaptive-5612019-evaluation',
                       'iter_2-convbert_adaptive-5972342-evaluation', 'iter_3-convbert_adaptive-5998181-evaluation',
                       'iter_4-convbert_adaptive-6057405-evaluation']
iter_names_uncertainty = ['iter_1-convbert_uncertainty-6200469-evaluation',
<<<<<<< HEAD
                         'iter_2-convbert_uncertainty-6253253-evaluation',
                         'iter_3-convbert_uncertainty-6318280-evaluation',
                          'iter_4-convbert_uncertainty-6423646-evaluation'
                         ]
=======
                          'iter_2-convbert_uncertainty-6253253-evaluation',
                          'iter_3-convbert_uncertainty-6318280-evaluation',
                          ]

new_iter_names = ['iter_0-convbert-969622-evaluation', 'iter_1-convbert_adaptive-5612019-evaluation',
                  'iter_2-convbert_adaptive-5972342-evaluation', 'iter_3-convbert_adaptive-5998181-evaluation',
                  'iter_4-convbert_adaptive-6057405-evaluation']

>>>>>>> dedbe0f6ba60739b5d03f267723d1a9a86d08bf5
ITERATIONS = 10
labels = ['no', 'yes']
