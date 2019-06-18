//
//  SpellChecker.swift
//  Misspelling
//
//  Created by Elia Cereda on 18/06/2019.
//  Copyright © 2019 Giorgia Adorni, Elia Cereda e Nassim Habbash. All rights reserved.
//

import Cocoa

class SpellChecker: NSObject, NSTextStorageDelegate {
    
    let linguisticTagger = NSLinguisticTagger(tagSchemes: [.tokenType], options: 0)
    
    unowned var textStorage: NSTextStorage? {
        willSet {
            self.textStorage?.delegate = nil
            self.linguisticTagger.string = nil
        }
        
        didSet {
            self.textStorage?.delegate = self
            self.linguisticTagger.string = self.textStorage?.mutableString as String?
        }
    }
    
    func configure(textView: NSTextView) {
        self.textStorage = textView.textStorage!
    }
    
    // MARK: - NSTextStorageDelegate
    func textStorage(_ textStorage: NSTextStorage, didProcessEditing editedMask: NSTextStorageEditActions, range editedRange: NSRange, changeInLength delta: Int) {
        self.linguisticTagger.stringEdited(in: editedRange, changeInLength: delta)
        self.performSpellCheck() // TODO: process editedRange + changeInLength to incrementally update the spell checking
    }
    
    // MARK: Spell Checking
    struct Token {
        var text: Substring
        var range: NSRange
    }
    
    struct CheckResult {
        var isMisspelled: Bool
    }
    
    private func performSpellCheck() {
        guard let text = self.linguisticTagger.string else {
            return
        }
        
        let tokens = self.tokenize(text: text)
        let results = self.spellCheck(tokens: tokens)
        
        DispatchQueue.main.async {
            self.updateLayoutManagers(tokens: tokens, results: results)
        }
    }
    
    private func tokenize(text: String) -> [Token] {
        var tokens = [Token]()
        
        let wholeText = NSRange(text.startIndex ..< text.endIndex, in: text)
        let options: NSLinguisticTagger.Options = [.omitWhitespace, .omitPunctuation, .omitOther]
        
        self.linguisticTagger.enumerateTags(in: wholeText, scheme: .tokenType, options: options) { (tag, range, enclosingRange, stop) in
            let swiftRange = Range(range, in: text)!
            let token = Token(text: text[swiftRange], range: range)
            
            tokens.append(token)
        }
        
        return tokens
    }
    
    private func spellCheck(tokens: [Token]) -> [CheckResult] {
        return tokens.map { CheckResult(isMisspelled: $0.text.count >= 5) }
    }
    
    private func updateLayoutManagers(tokens: [Token], results: [CheckResult]) {
        guard let layoutManagers = self.textStorage?.layoutManagers else {
            return
        }
        
        for layoutManager in layoutManagers {
            for (token, result) in zip(tokens, results) {
                if result.isMisspelled {
                    layoutManager.addTemporaryAttribute(.spellingState,
                                                        value: NSAttributedString.SpellingState.spelling.rawValue,
                                                        forCharacterRange: token.range)
                }
            }
        }
    }
}
