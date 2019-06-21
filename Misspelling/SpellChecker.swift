//
//  SpellChecker.swift
//  Misspelling
//
//  Created by Elia Cereda on 18/06/2019.
//  Copyright © 2019 Giorgia Adorni, Elia Cereda e Nassim Habbash. All rights reserved.
//

import Cocoa

class SpellChecker: NSObject, NSTextViewDelegate, NSTextStorageDelegate, NSPopoverDelegate, CandidatesViewController.Delegate {
    
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
    
    private var model = SpellCheckerModel()
    private var tokens = [Token]()
    private var results = [CheckResult]()
    
    private var candidatesList: NSPopover?
    private var candidatesViewController: CandidatesViewController? { return self.candidatesList?.contentViewController as? CandidatesViewController }
    private var isCandidatesListActive: Bool { return candidatesList != nil }
    private var isCandidatesListClosing: Bool = false
    private var lastActiveIndex: Int?
    
    private var transientEditRange: NSRange? = nil
    private var hasTransientEdit: Bool { return self.transientEditRange != nil }
    private var isApplyingTransientEdit: Bool = false
    
    // MARK: - NSTextViewDelegate
    func textView(_ textView: NSTextView, doCommandBy commandSelector: Selector) -> Bool {
        let event = NSApp.currentEvent
        
        switch commandSelector {
        case Selector(("noop:")) where event?.characters == "\0" && event?.modifierFlags.contains(.control) == true:
            self.toggleCandidatesList()
            return true
        case #selector(NSStandardKeyBindingResponding.cancelOperation(_:)) where self.isCandidatesListActive:
            self.hideCandidatesList(committingChanges: false)
            return true
        case #selector(NSStandardKeyBindingResponding.insertNewline(_:)) where self.isCandidatesListActive:
            self.hideCandidatesList(committingChanges: true)
            return true
            
        default:
            if let candidatesViewController = self.candidatesViewController {
                return candidatesViewController.doCommand(by: commandSelector)
            }
            return false
        }
    }
    
    func textViewDidChangeSelection(_ notification: Notification) {
        guard !self.isApplyingTransientEdit else {
            return
        }
        
        self.refreshCandidatesList()
    }
    
    func textShouldEndEditing(_ textObject: NSText) -> Bool {
        self.hideCandidatesList(committingChanges: true)
        return true
    }
    
    // MARK: - NSTextStorageDelegate
    func textStorage(_ textStorage: NSTextStorage, didProcessEditing editedMask: NSTextStorageEditActions, range editedRange: NSRange, changeInLength delta: Int) {
        guard !self.isApplyingTransientEdit else {
            return
        }
        
        self.hideCandidatesList(committingChanges: true)
        self.processEdits(in: editedRange, changeInLength: delta)
    }
    
    // MARK: Spell Checking
    enum Status {
        case invalid
        case inProgress
        case complete
    }
    
    struct Token: Hashable {
        var text: String
        var range: NSRange
    }
    
    struct Candidate: ConvertibleFromPython {
        var text: String
        var likelihood: Double
        
        init(text: String, likelihood: Double) {
            self.text = text
            self.likelihood = likelihood
        }
        
        init?(_ object: PythonObject) {
            guard let tuple = object.checking.tuple2,
                  let text = String(tuple.0),
                  let likelihood = Double(tuple.1) else {
                return nil
            }
            
            self.init(text: text, likelihood: likelihood)
        }
    }
    
    struct CheckResult {
        var isMispelled: Bool
        var candidates: [Candidate]
        
        init(isMispelled: Bool, candidates: [Candidate]) {
            self.isMispelled = isMispelled
            self.candidates = candidates
        }
    }
    
    private func processEdits(in editedRange: NSRange, changeInLength delta: Int) {
        self.linguisticTagger.stringEdited(in: editedRange, changeInLength: delta)
        
        self.invalidateSpellCheck()
        self.performSpellCheck() // TODO: process editedRange + changeInLength to incrementally update the spell checking
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
        
        self.results = self.model.spellCheck(tokens: self.tokens)
    }
    
    private func updateLayoutManagers() {
        assert(self.status == .inProgress)
        
        guard let layoutManagers = self.textStorage?.layoutManagers else {
            return
        }
        
        for layoutManager in layoutManagers {
            for (token, result) in zip(self.tokens, self.results) {
                if result.isMispelled {
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
            self.hideCandidatesList(committingChanges: true)
        }
    }
    
    func showCandidatesList() {
        guard !self.isCandidatesListActive else {
            return
        }
        
        let viewController = CandidatesViewController()
        
        viewController.delegate = self
        
        let popover = NSPopover()
        
        popover.delegate = self
//        popover.behavior = .transient
        popover.contentViewController = viewController
        popover.show(relativeTo: self.textView!.bounds, of: self.textView!, preferredEdge: .minY)
        
        self.candidatesList = popover
        self.refreshCandidatesList()
    }
    
    func hideCandidatesList(committingChanges: Bool) {
        guard self.isCandidatesListActive else {
            return
        }
        
        if committingChanges {
            self.commitTransientEdits()
        }
        
        self.candidatesList!.performClose(self)
    }
    
    // MARK: - NSPopoverDelegate
    
    func popoverWillClose(_ notification: Notification) {
        // Any confirmed edits should be committed by now
        self.cancelTransientEdits()
        
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
        
        let activeIndex: Int?
        
        if self.hasTransientEdit {
            if self.insertionPointIsInsideTransientEdit() {
                // The insertion point is still inside the range of the transient edit.
                // The active index has not changed.
                activeIndex = self.lastActiveIndex
            } else {
                // The insertion point has left the range of the transient edit.
                // Commit the changes, triggering a new spell check, and return.
                // When spell checking completes it will call refreshCandidatesList() again.
                self.commitTransientEdits()
                return
            }
        } else {
            activeIndex = self.indexOfActiveWord()
        }
        
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
    
    private func insertionPointRange() -> NSRange? {
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
        
        return first
    }
    
    private func indexOfActiveWord() -> Int? {
        guard let insertionPoint = self.insertionPointRange() else {
            return nil
        }
        
        // FIXME: tokens is sorted by range, could use binary search.
        let wordIndex = self.tokens.firstIndex {
            self.range(insertionPoint, isInside: $0.range)
        }
        
        return wordIndex
    }
    
    private func insertionPointIsInsideTransientEdit() -> Bool {
        guard let insertionPoint = self.insertionPointRange(),
              let transientEditRange = self.transientEditRange else {
            return false
        }
        
        return self.range(insertionPoint, isInside: transientEditRange)
    }
    
    private func range(_ range: NSRange, isInside other: NSRange) -> Bool {
        return other.intersection(range) != nil || range.lowerBound == other.upperBound
    }
    
    // MARK: - CandidatesViewController.Delegate
    func candidatesViewControllerSelectionDidChange(_ viewController: CandidatesViewController) {
        guard let activeIndex = self.lastActiveIndex else {
            return
        }
        
        let editRange = self.transientEditRange ?? self.tokens[activeIndex].range
        
        guard let selectedCandidate = viewController.selectedCandidate else {
            // Should not happen, one candidate should always be selected
            return
        }
        
        let candidate = self.results[activeIndex].candidates[selectedCandidate].text
        let newRange = NSRange(location: editRange.location, length: (candidate as NSString).length)
        
        self.transientEditRange = newRange
        
        self.isApplyingTransientEdit = true
        self.textStorage?.replaceCharacters(in: editRange, with: candidate)
        self.isApplyingTransientEdit = false
    }
    
    func cancelTransientEdits() {
        guard let editRange = self.transientEditRange,
              let activeIndex = self.lastActiveIndex else {
            return
        }
        
        let originalText = self.tokens[activeIndex].text
        
        self.transientEditRange = nil
        
        self.isApplyingTransientEdit = true
        self.textStorage?.replaceCharacters(in: editRange, with: originalText)
        self.isApplyingTransientEdit = true
    }
    
    func commitTransientEdits() {
        guard let editRange = self.transientEditRange,
              let activeIndex = self.lastActiveIndex else {
            return
        }
        
        self.transientEditRange = nil
        
        let originalRange = self.tokens[activeIndex].range
        
        // Re-send the event that was suppressed when the transient edit was first made
        self.processEdits(in: originalRange, changeInLength: editRange.length - originalRange.length)
    }
}
