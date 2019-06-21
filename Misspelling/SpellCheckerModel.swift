//
//  SpellCheckerModel.swift
//  Misspelling
//
//  Created by Elia Cereda on 21/06/2019.
//  Copyright © 2019 Giorgia Adorni, Elia Cereda e Nassim Habbash. All rights reserved.
//

import Foundation

private let hmm = Python.import("hmm")

class SpellCheckerModel {
    
    typealias Token = SpellChecker.Token
    typealias CheckResult = SpellChecker.CheckResult
    typealias Candidate = SpellChecker.Candidate
    
    private var model: PythonObject!
    
    private var cache = [Token: CheckResult]()
    private let queue = DispatchQueue(label: "SpellCheckerModel", qos: .userInteractive)
    
    init() {
        self.queue.async {
            var time: Double
                
            time = measure {
                _ = hmm.pp
            }
            
            print("Python init: \(time) ms")
            
//            time = measure {
//                self.model = hmm.HMM(order: 1, max_edits: 2, max_states: 3)
//            }
//
//            print("Constructor: \(time) ms")
//
//            time = measure {
//                self.model.train(
//                    words_ds: Bundle.main.path(forResource: "data/word_freq/frequency-alpha-gcide", ofType: "txt")!,
//                    sentences_ds: Bundle.main.path(forResource: "data/texts/big", ofType: "txt")!,
//                    typo_ds: Bundle.main.path(forResource: "data/typo/new/train", ofType: "csv")!
//                )
//            }
//
//            print("Training: \(time) ms")
            
            time = measure {
                self.model = hmm.HMM.load(
                    file: Bundle.main.path(forResource: "hmm", ofType: "pickle")!
                )
            }
            
            print("Load: \(time) ms")
        }
    }
    
    func spellCheck(tokens: [Token]) -> [CheckResult] {
        var results: [CheckResult]!
        
        let time = measure {
            self.queue.sync {
                results = tokens.map {
                    var result = self.cache[$0]
                    
                    if result == nil {
                        let candidates = [Candidate](self.model.candidates(word: $0.text)) ?? []
                        let isMispelled = ($0.text != candidates.first?.text)
                    
                        result = CheckResult(isMispelled: isMispelled,
                                             candidates: candidates)
                        
                        self.cache[$0] = result
                    }
                    
                    return result!
                }
            }
        }
        
        print("Spell Check: \(time) ms")
        
        return results
    }
}
