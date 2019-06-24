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
                results = tokens.map { token in
                    var result = self.cache[token]
                    
                    if result == nil {
                        let candidates = [Candidate](self.model.candidates(word: token.text, max_states: 10)) ?? []
                        let isMispelled = candidates.allSatisfy { token.text != $0.text }
                    
                        result = CheckResult(isMispelled: isMispelled,
                                             candidates: self.normalized(candidates))
                        
                        self.cache[token] = result
                    }
                    
                    return result!
                }
            }
        }
        
        print("Spell Check: \(time) ms")
        
        return results
    }
    
    private func normalized(_ candidates: [Candidate]) -> [Candidate] {
        let total = candidates.map { $0.likelihood }.reduce(0, +)
        
        return candidates.map {
            Candidate(text: $0.text, likelihood: $0.likelihood / total)
        }
    }
}
