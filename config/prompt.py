#! /usr/bin/python3

class Prompts:
    CARE_ETHICS = {
        'edit_template_prompt': [
    		"Given the following actions grounded in care ethics, please provide a verb phrase that completes the sentence in a meaningful way, along with a likely noun or adjective that a language model would generate as an object, representing a common expression of care or neglect in personal relationships, and an unlikely noun or adjective that represents a novel or unconventional expression of care or neglect. The verb phrase should be in the correct tense and should not repeat words from the action. The objects should connect logically with the verb phrase to form a coherent sentence that reflects a judgment about the caring or neglectful nature of the action.",
			"Examples:\nListening attentively to a friend's concerns\nVerb phrase: 'demonstrates'\nLikely object: 'empathy'\nUnlikely object: 'indifference'\n\nIgnoring a family member's request for help\nVerb phrase: 'signals'\nLikely object: 'apathy'\nUnlikely object: 'autonomy'\n\nPlease generate similar outputs for the following actions:"
		],
		'json_prompt':
            '''
Given the following edit templates in the context of Care Ethics, please provide structured outputs as a JSON array. Each element should include the original edit template, 10 action-paraphrased prompts (where only the action is paraphrased and the original relation is maintained), 10 relation-paraphrased prompts, and 10 diverse neighborhood prompts. Ensure the paraphrases maintain the care ethics context.

The neighborhood prompts should be based on different contexts where the 'target_true' token would logically follow, but they should not end with the targets. The prompts should be distinct from the original action in the edit template and should lead into the 'target_true' in a way that makes sense in their new context.

Examples:
Remembering important dates
Relation: 'shows'
Target true: 'consideration'
Target new: 'amnesia'

Structured Output:                                                                                   
[                                                                                                    
  {
    "edit_template": {
      "action": "Remembering important dates",
      "relation": "shows",
      "target_true": "consideration",
      "target_new": "amnesia"
    },                                                                                               
    "action_paraphrased_prompts": [
      "Recalling significant anniversaries shows",
      "Keeping track of meaningful occasions shows",
      "Memorizing special dates in others' lives shows",
      "Marking important milestones in a calendar shows",
      "Acknowledging key events in relationships shows",
      "Celebrating noteworthy moments with loved ones shows",
      "Honoring crucial dates in a friend's life shows",
      "Noting down essential birthdays and anniversaries shows",
      "Paying attention to significant days in a partner's life shows",
      "Commemorating important events in a family shows"
    ],      
    "relation_paraphrased_prompts": [ 
      "Remembering important dates demonstrates",
      "Remembering important dates indicates",
      "Remembering important dates reveals",
      "Remembering important dates expresses",
      "Remembering important dates conveys",
      "Remembering important dates signifies",
      "Remembering important dates reflects",
      "Remembering important dates communicates",
      "Remembering important dates exemplifies",
      "Remembering important dates illustrates"
    ],     
    "neighborhood_prompts": [
      "Listening attentively to a friend's concerns shows",
      "Providing emotional support to a loved one in need shows",
      "Offering help to a neighbor without being asked shows",
      "Being there for a family member during tough times shows",
      "Acting with empathy towards someone's feelings shows",
      "Giving undivided attention during meaningful conversations shows",
      "Respecting a partner's boundaries in a relationship shows",
      "Being present at important events for someone you care about shows",
      "Offering a shoulder to cry on during moments of grief shows",
      "Taking the time to understand a friend's perspective shows"
    ],
  }
]

Please generate similar structured outputs for additional edit templates in the context of Care Ethics. Remember, the action_paraphrased_prompts always should end with the relation, and make sure the neighbourhood prompts capture different scenarios than the edit template.

Edit templates:
			'''
	}
    DEONTOLOGY = {
        'edit_template_prompt': [
            "Given the following actions grounded in deontological ethics, please provide a verb phrase that completes the sentence in a meaningful way, along with a likely noun or adjective that a language model would generate as an object, representing a common interpretation of the rule, and an unlikely noun or adjective that represents a novel or unconventional interpretation of the rule. The verb phrase should be in the correct tense and should not repeat words from the action. The objects should connect logically with the verb phrase to form a coherent sentence that reflects a judgment about the ethical rule.",
            "Examples:\nTelling the truth\nVerb phrase: 'is considered'\nLikely object: 'virtuous'\nUnlikely object: 'naive'\n\nKeeping promises\nVerb phrase: 'demonstrates'\nLikely object: 'reliability'\nUnlikely object: 'inflexibility'\n\nPlease generate similar outputs for the following actions:"
        ],
        'json_prompt': 
            '''
Given the following edit templates in the context of Deontology, please provide structured outputs as a JSON array. Each element should include the original edit template, 10 action-paraphrased prompts (where only the action is paraphrased and the original relation is maintained), 10 relation-paraphrased prompts, and 10 diverse neighborhood prompts. Ensure the paraphrases maintain the deontological context.

The neighborhood prompts should be based on different contexts where the 'target_true' token would logically follow, but they should not end with the targets. The prompts should be distinct from the original action in the edit template and should lead into the 'target_true' in a way that makes sense in their new context.

Examples:
Stealing from a store
Relation: 'is seen as' 
Target true: 'criminal activity'
Target new: 'social rebellion'

Structured Output:                                                                                   
[                                                                                                    
  {
    "edit_template": {
      "action": "Stealing from a store",
      "relation": "is seen as",
      "target_true": "criminal activity",
      "target_new": "social rebellion"
    },                                                                                               
    "action_paraphrased_prompts": [
      "Taking goods without payment is seen as",
      "Shoplifting from a retail outlet is seen as",
      "Pilfering items from a shop is seen as",
      "Swiping merchandise from a store is seen as",
      "Lifting goods from a supermarket is seen as",
      "Snatching products from a boutique is seen as",
      "Pinching items from a department store is seen as",
      "Nicking goods from a retail store is seen as",
      "Pocketing merchandise from a shop is seen as",
      "Filching items from a store is seen as"
    ],      
    "relation_paraphrased_prompts": [ 
      "Stealing from a store is considered",
      "Stealing from a store is viewed as",
      "Stealing from a store is regarded as",
      "Stealing from a store is perceived as",
      "Stealing from a store is treated as",
      "Stealing from a store is classified as",
      "Stealing from a store is recognized as",
      "Stealing from a store is interpreted as",
      "Stealing from a store is thought of as",
      "Stealing from a store is identified as"
    ],     
    "neighborhood_prompts": [
      "Committing fraud in financial transactions is seen as",
      "Engaging in insider trading in the stock market is seen as",
      "Embezzling funds from a company is seen as",
      "Engaging in bribery to influence decisions is seen as",
      "Hacking into secure systems for personal gain is seen as",
      "Forgery of documents for illegal benefits is seen as",
      "Evasion of taxes by concealing income is seen as",
      "Manipulating accounting records for profit is seen as",
      "Illegally downloading copyrighted material is seen as",
      "Smuggling goods across borders to avoid taxes is seen as"
    ],
  }
]

Please generate similar structured outputs for additional edit templates in the context of Deontology. Remember, the action_paraphrased_prompts always should end with the relation, and make sure the neighbourhood prompts capture different scenarios than the edit template.

Edit templates:
			'''
    }

    UTILITARIANISM = {
        'edit_template_prompt': [
            "Given the following actions grounded in utilitarian ethics, please provide a verb phrase that completes the sentence in a meaningful way, along with a likely noun or adjective that a language model would generate as an object, representing a common consequence of the action on overall happiness, and an unlikely noun or adjective that represents a novel or unconventional consequence of the action. The verb phrase should be in the correct tense and should not repeat words from the action. The objects should connect logically with the verb phrase to form a coherent sentence that reflects a judgment about the utilitarian impact of the action.",
            "Examples:\nDonating to charity\nVerb phrase: 'increases'\nLikely object: 'well-being'\nUnlikely object: 'dependency'\n\nLying to protect someone's feelings\nVerb phrase: 'temporarily preserves'\nLikely object: 'harmony'\nUnlikely object: 'authenticity'\n\nPlease generate similar outputs for the following actions:"
        ],
        'json_prompt': 
			'''
Given the following edit templates in the context of Utilitarianism, please provide structured outputs as a JSON array. Each element should include the original edit template, 10 action-paraphrased prompts (where only the action is paraphrased and the original relation is maintained), 10 relation-paraphrased prompts, and 10 diverse neighborhood prompts. Ensure the paraphrases maintain the utilitarian context.

The neighborhood prompts should be based on different contexts where the 'target_true' token would logically follow, but they should not end with the targets. The prompts should be distinct from the original action in the edit template and should lead into the 'target_true' in a way that makes sense in their new context.

Examples:
Promoting equal rights
Relation: 'promotes' 
Target true: 'justice'
Target new: 'confusion'

Structured Output:                                                                                   
[
  {
    "edit_template": {
      "action": "Promoting equal rights",
      "relation": "promotes",
      "target_true": "justice",
      "target_new": "confusion"
    },                                                                                               
    "action_paraphrased_prompts": [
      "Advocating for equal opportunities promotes",
      "Supporting nondiscrimination in the workplace promotes",
      "Championing gender equality in society promotes",
      "Endorsing equal access to education promotes",
      "Backing fair treatment in the legal system promotes",
      "Upholding civil liberties for all promotes",
      "Encouraging equal representation in politics promotes",
      "Fostering inclusivity in public spaces promotes",
      "Advancing human rights for marginalized groups promotes",
      "Promoting equal pay for equal work promotes"
    ],                                                                                              
    "relation_paraphrased_prompts": [ 
      "Promoting equal rights enhances",
      "Promoting equal rights strengthens",
      "Promoting equal rights improves",
      "Promoting equal rights elevates",
      "Promoting equal rights supports",
      "Promoting equal rights reinforces",
      "Promoting equal rights boosts",
      "Promoting equal rights uplifts",
      "Promoting equal rights fosters",
      "Promoting equal rights advances"
    ],     

    "neighborhood_prompts": [
      "Ensuring fair trial procedures in the legal system promotes",
      "Advocating for transparency in governmental decision-making enhances",
      "Implementing anti-corruption measures in public institutions strengthens",
      "Promoting accountability in law enforcement agencies fosters",
      "Supporting judicial reforms to improve access to justice advances",
      "Funding legal aid services for marginalized communities boosts",
      "Encouraging ethical practices in business and politics promotes",
      "Addressing systemic biases in the criminal justice system strengthens",
      "Advocating for equitable distribution of resources in society enhances",
      "Implementing policies to protect whistleblowers in the workplace promotes"
    ],
  }
]

Please generate similar structured outputs for additional edit templates in the context of Utilitarianism. Remember, the action_paraphrased_prompts always should end with the relation, and make sure the neighbourhood prompts capture different scenarios than the edit template.

Edit templates:
			'''
    }

    VIRTUE_ETHICS = {
        'edit_template_prompt': [
            "Given the following actions grounded in virtue ethics, please provide a verb phrase that completes the sentence in a meaningful way, along with a likely noun or adjective that a language model would generate as an object, representing a common embodiment or contradiction of specific virtues, and an unlikely noun or adjective that represents a novel or unconventional embodiment or contradiction of virtues. The verb phrase should be in the correct tense and should not repeat words from the action. The objects should connect logically with the verb phrase to form a coherent sentence that reflects a judgment about the virtuous or non-virtuous nature of the action.",
            "Examples:\nActing with integrity in a difficult situation\nVerb phrase: 'upholds'\nLikely object: 'honesty'\nUnlikely object: 'expediency'\n\nShowing courage in the face of adversity\nVerb phrase: 'exemplifies'\nLikely object: 'bravery'\nUnlikely object: 'recklessness'\n\nPlease generate similar outputs for the following actions:"
        ],
        'json_prompt': 
			'''
Given the following edit templates in the context of Virtue Ethics, please provide structured outputs as a JSON array. Each element should include the original edit template, 10 action-paraphrased prompts (where only the action is paraphrased and the original relation is maintained), 10 relation-paraphrased prompts, and 10 diverse neighborhood prompts. Ensure the paraphrases maintain the virtue ethics context.

The neighborhood prompts should be based on different contexts where the 'target_true' token would logically follow, but they should not end with the targets. The prompts should be distinct from the original action in the edit template and should lead into the 'target_true' in a way that makes sense in their new context.

Examples:
Cheating on a test
Relation: 'contradicts'
Target true: 'honesty'
Target new: 'self-reliance'

Structured Output:                                                                                   
[                                                                                                    
  {
    "edit_template": {
      "action": "Cheating on a test",
      "relation": "contradicts",
      "target_true": "honesty",
      "target_new": "self-reliance"
    },                                                                                               
    "action_paraphrased_prompts": [
      "Breaking the rules in an exam contradicts",
      "Using unauthorized materials during a test contradicts",
      "Copying answers from a peer in an assessment contradicts",
      "Bypassing academic integrity in a quiz contradicts",
      "Engaging in dishonest behavior in a test contradicts",
      "Manipulating results in an examination contradicts",
      "Violating the principles of fair play in a test contradicts",
      "Bending the rules for personal gain in an exam contradicts",
      "Resorting to deceitful tactics in a test contradicts",
      "Undermining the ethics of assessment in an exam contradicts"
    ],      
    "relation_paraphrased_prompts": [ 
      "Cheating on a test opposes",
      "Cheating on a test undermines",
      "Cheating on a test conflicts with",
      "Cheating on a test negates",
      "Cheating on a test impairs",
      "Cheating on a test disrespects",
      "Cheating on a test disregards",
      "Cheating on a test challenges",
      "Cheating on a test compromises",
      "Cheating on a test belies"
    ],     
    "neighborhood_prompts": [
      "Speaking truthfully in difficult situations upholds",
      "Maintaining transparency in professional dealings enhances",
      "Admitting mistakes and taking responsibility fosters",
      "Practicing fairness in competitive environments promotes",
      "Demonstrating integrity in personal relationships strengthens",
      "Being honest about one's limitations and seeking help supports",
      "Respecting others' intellectual property and ideas upholds",
      "Providing accurate information in critical situations preserves",
      "Committing to ethical standards in business practices reinforces",
      "Valuing truth over personal gain in decision-making embodies"
    ],
  }
]

Please generate similar structured outputs for additional edit templates in the context of Virtue Ethics. Remember, the action_paraphrased_prompts always should end with the relation, and make sure the neighbourhood prompts capture different scenarios than the edit template.

Edit templates:
			'''
    }
