//
//  SpellChecker.swift
//  Misspelling
//
//  Created by Elia Cereda on 18/06/2019.
//  Copyright © 2019 Giorgia Adorni, Elia Cereda e Nassim Habbash. All rights reserved.
//

import Cocoa

class SpellChecker: NSObject, NSTextViewDelegate, NSTextStorageDelegate, NSPopoverDelegate {
    
    let linguisticTagger = NSLinguisticTagger(tagSchemes: [.tokenType], options: 0)
    
    unowned var textView: NSTextView? {
        willSet {
            self.textView?.delegate = nil
            self.textStorage = nil
        }
        
        didSet {
            self.textView?.delegate = self
            self.textStorage = self.textView?.textStorage!
        }
    }
    
    private unowned var textStorage: NSTextStorage? {
        willSet {
            self.textStorage?.delegate = nil
            self.linguisticTagger.string = nil
        }
        
        didSet {
            self.textStorage?.delegate = self
            self.linguisticTagger.string = self.textStorage?.mutableString as String?
        }
    }
    
    private var status = Status.invalid
    private var invalidationCounter = 0
    
    private var tokens = [Token]()
    private var results = [CheckResult]()
    
    private var candidatesList: NSPopover?
    private var candidatesViewController: CandidatesViewController? { return self.candidatesList?.contentViewController as? CandidatesViewController }
    private var isCandidatesListActive: Bool { return candidatesList != nil }
    private var isCandidatesListClosing: Bool = false
    private var lastActiveIndex: Int?
    
    // MARK: - NSTextViewDelegate
    func textView(_ textView: NSTextView, doCommandBy commandSelector: Selector) -> Bool {
        let event = NSApp.currentEvent
        
        switch commandSelector {
        case Selector(("noop:")) where event?.characters == "\0" && event?.modifierFlags.contains(.control) == true:
            self.toggleCandidatesList()
            return true
        case Selector(("cancel:")) where self.isCandidatesListActive:
            self.hideCandidatesList()
            return true
        case Selector(("insertNewline:")) where self.isCandidatesListActive:
            self.commitSelectedCandidate()
            return true
            
        default:
            if let candidatesViewController = self.candidatesViewController {
                return candidatesViewController.doCommand(by: commandSelector)
            }
            return false
        }
    }
    
    func textView(_ textView: NSTextView, willChangeSelectionFromCharacterRanges oldSelectedCharRanges: [NSValue], toCharacterRanges newSelectedCharRanges: [NSValue]) -> [NSValue] {
        return newSelectedCharRanges
    }
    
    func textViewDidChangeSelection(_ notification: Notification) {
        self.refreshCandidatesList()
    }
    
    func textShouldEndEditing(_ textObject: NSText) -> Bool {
        self.hideCandidatesList()
        return true
    }
    
    // MARK: - NSTextStorageDelegate
    func textStorage(_ textStorage: NSTextStorage, didProcessEditing editedMask: NSTextStorageEditActions, range editedRange: NSRange, changeInLength delta: Int) {
        self.linguisticTagger.stringEdited(in: editedRange, changeInLength: delta)
        
        self.hideCandidatesList()
        
        self.invalidateSpellCheck()
        self.performSpellCheck() // TODO: process editedRange + changeInLength to incrementally update the spell checking
    }
    
    // MARK: Spell Checking
    enum Status {
        case invalid
        case inProgress
        case complete
    }
    
    struct Token {
        var text: String
        var range: NSRange
    }
    
    struct Candidate {
        var text: String
    }
    
    struct CheckResult {
        var isMisspelled: Bool
        var candidates: [Candidate]
    }
    
    private func invalidateSpellCheck() {
        self.status = .invalid
        self.invalidationCounter += 1
        
        self.tokens.removeAll(keepingCapacity: true)
        self.results.removeAll(keepingCapacity: true)
    }
    
    private func performSpellCheck() {
        guard self.status == .invalid else {
            return
        }
        
        self.status = .inProgress
        
        guard let text = self.linguisticTagger.string else {
            return
        }
        
        self.tokenize(text: text)
        self.spellCheck()
        
        let invalidationCounter = self.invalidationCounter
        
        DispatchQueue.main.async {
            guard self.invalidationCounter == invalidationCounter else {
                // This spell check request was canceled, ignore it.
                return
            }
            
            self.updateLayoutManagers()
            self.refreshCandidatesList()
            self.status = .complete
        }
    }
    
    private func tokenize(text: String) {
        assert(self.status == .inProgress)
        
        let wholeText = NSRange(text.startIndex ..< text.endIndex, in: text)
        let options: NSLinguisticTagger.Options = [.omitWhitespace, .omitPunctuation, .omitOther]
        
        self.linguisticTagger.enumerateTags(in: wholeText, scheme: .tokenType, options: options) { (tag, range, sentenceRange, stop) in
            let swiftRange = Range(range, in: text)!
            let tokenText = text[swiftRange]
            
            let token = Token(text: String(tokenText), range: range)
            
            self.tokens.append(token)
        }
    }
    
    private func spellCheck() {
        assert(self.status == .inProgress)
        
        self.results = self.tokens.map {
            CheckResult(isMisspelled: $0.text.count >= 5, candidates: [
                Candidate(text: "prova"), Candidate(text: "alcune"), Candidate(text: "alternative"),
                Candidate(text: "prova"), Candidate(text: "alcune"), Candidate(text: "alternative"),
            ])
        }
    }
    
    private func updateLayoutManagers() {
        assert(self.status == .inProgress)
        
        guard let layoutManagers = self.textStorage?.layoutManagers else {
            return
        }
        
        for layoutManager in layoutManagers {
            for (token, result) in zip(self.tokens, self.results) {
                if result.isMisspelled {
                    layoutManager.addTemporaryAttribute(.spellingState,
                                                        value: NSAttributedString.SpellingState.spelling.rawValue,
                                                        forCharacterRange: token.range)
                }
            }
        }
    }
    
    // MASK: - Candidates User Interface
    func toggleCandidatesList() {
        self.performSpellCheck()
        
        if self.candidatesList == nil {
            self.showCandidatesList()
        } else {
            self.hideCandidatesList()
        }
    }
    
    func showCandidatesList() {
        guard !self.isCandidatesListActive else {
            return
        }
        
        let popover = NSPopover()
        
        popover.delegate = self
//        popover.behavior = .transient
        popover.contentViewController = CandidatesViewController()
        popover.show(relativeTo: self.textView!.bounds, of: self.textView!, preferredEdge: .minY)
        
        self.candidatesList = popover
        self.refreshCandidatesList()
    }
    
    func hideCandidatesList() {
        guard self.isCandidatesListActive else {
            return
        }
        
        self.candidatesList!.performClose(self)
    }
    
    func commitSelectedCandidate() {
        // TODO
        self.hideCandidatesList()
    }
    
    // MARK: - NSPopoverDelegate
    
    func popoverWillClose(_ notification: Notification) {
        self.isCandidatesListClosing = true
    }
    
    func popoverDidClose(_ notification: Notification) {
        self.candidatesList = nil
        self.lastActiveIndex = nil
        self.isCandidatesListClosing = false
    }
    
    private func refreshCandidatesList() {
        guard let textView = self.textView,
              let candidatesList = self.candidatesList,
              let candidatesViewController = self.candidatesViewController else {
            return
        }
        
        // Prevent glitches if a popover is refreshed while it's being closed
        guard !self.isCandidatesListClosing else {
            return
        }
        
        let activeIndex = self.indexOfActiveWord()
        
        guard activeIndex != self.lastActiveIndex || activeIndex == nil else {
            return
        }
        
        self.lastActiveIndex = activeIndex
        
        if let activeIndex = activeIndex {
            let activeWord = self.tokens[activeIndex]
            let activeResult = self.results[activeIndex]
        
            let wordRange = activeWord.range
            let wordRect = textView.firstRect(forCharacterRange: wordRange, actualRange: nil)
        
            let windowRect = textView.window!.convertFromScreen(wordRect)
            let viewRect = textView.convert(windowRect, from: nil)
        
            candidatesList.positioningRect = viewRect
            candidatesViewController.candidates = activeResult.candidates
            candidatesViewController.selectedCandidate = activeResult.candidates.firstIndex { $0.text == activeWord.text }
        } else {
            let insertionPoint = textView.firstRect(forCharacterRange: textView.selectedRange(), actualRange: nil)
            let windowRect = textView.window!.convertFromScreen(insertionPoint)
            var viewRect = textView.convert(windowRect, from: nil)
            
            assert(viewRect.width == 0)
            viewRect.origin.x -= 0.5
            viewRect.size.width = 1
            
            candidatesList.positioningRect = viewRect
            candidatesViewController.candidates = nil
            candidatesViewController.selectedCandidate = nil
        }
    }
    
    private func indexOfActiveWord() -> Int? {
        guard let textView = self.textView else {
                return nil
        }
        
        let selectedRanges = textView.selectedRanges
        
        guard var first = selectedRanges.first?.rangeValue else {
            return nil
        }
        
        if selectedRanges.count > 1 || first.length > 0 {
            let location: Int
            
            if textView.selectionAffinity == .upstream {
                location = first.lowerBound
            } else {
                location = first.upperBound
            }
            
            first.location = location
            first.length = 0
            
            textView.setSelectedRange(first, affinity: textView.selectionAffinity, stillSelecting: false)
        }
        
        // FIXME: tokens is sorted by range, could use binary search.
        let firstWord = self.tokens.firstIndex {
            ($0.range.intersection(first) != nil) ||  (first.length == 0 && $0.range.upperBound == first.lowerBound)
        }
        
        return firstWord
    }
}
