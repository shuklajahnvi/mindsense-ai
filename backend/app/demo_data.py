"""Synthetic demo journal entries.

These texts were written by hand for demonstration purposes (a fictional
university student over ~6 weeks). They are NOT real user data. When seeded,
each entry is analysed by the actual trained models - predictions shown for
demo entries are genuine model outputs on this synthetic text.
"""

# (days_ago, hour, mood, energy, sleep_hours, sleep_quality, text)
DEMO_ENTRIES = [
    (42, 21, 4, 4, 7.5, 4, "First week of the new term. I'm actually excited about my machine learning module and the lecturer seems great. Had dinner with Priya and Sam and we laughed a lot."),
    (41, 22, 4, 3, 7.0, 4, "Went for a long walk by the river after classes. It felt calming to be outside. Grateful that the weather held up."),
    (39, 23, 3, 3, 6.0, 3, "The reading list is longer than I expected. Not sure how I'm going to keep up with everything alongside my shifts at the cafe."),
    (38, 20, 4, 4, 7.0, 4, "Joined the climbing club! I was nervous walking in but everyone was friendly. I'm proud I actually went."),
    (36, 22, 3, 3, 6.5, 3, "Work was busy and my manager asked me to cover an extra shift this weekend. I said yes but now I'm a bit annoyed at myself."),
    (35, 21, 4, 4, 7.5, 4, "Mum called and we talked for an hour. I miss home sometimes but it was lovely to hear her voice."),
    (33, 0, 2, 2, 5.0, 2, "Couldn't sleep again. Kept thinking about the group project, two people haven't replied to messages in days and the deadline is next Friday."),
    (32, 22, 2, 2, 5.5, 2, "Stressed about the group project. I ended up doing most of the analysis myself and I'm frustrated that it doesn't feel fair."),
    (31, 21, 3, 3, 6.5, 3, "We finally had a meeting and divided up the remaining work. Relieved that it's moving, though I'm still worried about the timeline."),
    (29, 23, 3, 2, 6.0, 3, "Long shift at the cafe, my feet hurt. Too tired to study, which makes me anxious about falling behind."),
    (28, 19, 4, 4, 8.0, 4, "Slept properly for once and it made such a difference. Climbed in the evening and finally finished the route I'd been stuck on."),
    (26, 22, 3, 3, 6.5, 3, "Submitted the group project. It's not perfect but it's done. Mostly feeling relief and a bit of exhaustion."),
    (24, 21, 4, 4, 7.0, 4, "Got good feedback on my last essay. The tutor said my argument was clear. That made my week honestly."),
    (22, 22, 3, 3, 6.5, 3, "Exam timetable came out. Three exams in five days. Trying not to panic and to make a revision plan instead."),
    (21, 23, 2, 2, 5.5, 2, "Revision isn't going well. I keep re-reading the same notes and nothing sticks. I feel like everyone else is ahead of me."),
    (20, 1, 2, 2, 4.5, 2, "It's past 1am and I'm still awake worrying about statistics. My chest feels tight when I think about the exam."),
    (19, 21, 3, 3, 6.0, 3, "Talked to my tutor about how stressed I've been. She suggested breaking revision into small blocks and taking proper breaks. It helped to say it out loud."),
    (18, 22, 3, 3, 6.5, 3, "Tried the study blocks with breaks. Got through two past papers. Still nervous but it feels a bit more manageable."),
    (17, 22, 2, 2, 5.0, 2, "Didn't go to climbing because I felt guilty about not studying. Ate dinner at my desk. Everything feels like pressure right now."),
    (15, 20, 3, 3, 6.0, 3, "First exam today. Some questions were harder than I expected and I'm disappointed with a couple of my answers, but I think I passed."),
    (14, 21, 3, 2, 6.0, 3, "Second exam done. So tired. Priya brought me snacks and we revised together for the last one, which was kind of her."),
    (13, 22, 4, 3, 7.0, 4, "Last exam finished! Massive relief. Went out for pizza with friends and laughed about how stressed we all were."),
    (11, 11, 4, 4, 9.0, 4, "Slept nine hours. Spent the morning doing nothing and it felt amazing. Going to the climbing wall later."),
    (10, 20, 4, 4, 8.0, 4, "Visited home for the weekend. Walked the dog with my brother and helped mum in the garden. Feeling calm and grateful."),
    (8, 21, 3, 3, 7.0, 3, "Back at uni. Starting to think about my dissertation topic, a bit unsure which direction to choose but curious about a few ideas."),
    (6, 22, 4, 4, 7.5, 4, "Met my potential supervisor and she was really encouraging about my idea on analysing journaling apps. Excited to start."),
    (4, 22, 3, 3, 6.5, 3, "The cafe changed my hours without asking. Annoyed, and money is a bit tight this month, which worries me."),
    (3, 21, 4, 4, 7.5, 4, "Climbing club social tonight. I'm glad I kept going this term, these people have become good friends."),
    (1, 22, 3, 3, 6.5, 3, "Busy week ahead with a presentation on Thursday. Feeling a mix of nervous and prepared. I'll practise it tomorrow with Sam."),
]
