//
//  CandidatesViewController.swift
//  Misspelling
//
//  Created by Elia Cereda on 18/06/2019.
//  Copyright © 2019 Giorgia Adorni, Elia Cereda e Nassim Habbash. All rights reserved.
//

import Cocoa

infix operator %%: MultiplicationPrecedence

extension Int {
    static func %% (lhs: Int, rhs: Int) -> Int {
        let r = lhs % rhs
        
        return (r >= 0) ? r : r + rhs
    }
}

private extension NSUserInterfaceItemIdentifier {
    static let candidateCell = NSUserInterfaceItemIdentifier("CandidateCell")
}

class CandidatesViewController: NSViewController, NSTableViewDataSource, NSTableViewDelegate {

    var hasContent: Bool { return (self.candidates?.count ?? 0) != 0 }
    
    var candidates: [SpellChecker.Candidate]? {
        didSet {
            self.tableView.reloadData()
            self.tableView.isHidden = !self.hasContent
            
            self.noContentLabel.isHidden = self.hasContent
            
            self.preferredContentSize = self.tableView.sizeThatFits(NSSize(width: 100, height: 0))
        }
    }
    
    var selectedCandidate: Int? {
        get { return (self.tableView.selectedRow != -1) ? self.tableView.selectedRow : nil }
        set {
            if let newValue = newValue {
                self.tableView.selectRowIndexes([newValue], byExtendingSelection: false)
            } else {
                self.tableView.deselectAll(self)
            }
        }
    }
    
    @IBOutlet weak var tableView: NSTableView!
    @IBOutlet weak var noContentLabel: NSTextField!
    
    override func viewDidLoad() {
        super.viewDidLoad()
        // Do view setup here.
    }
    
    // MARK: Commands
    
    func doCommand(by selector: Selector) -> Bool {
        switch selector {
        case #selector(NSStandardKeyBindingResponding.moveUp(_:)),
             #selector(NSStandardKeyBindingResponding.moveDown(_:)):
            
            self.perform(selector, with: self)
            return true
            
        default:
            return false
        }
    }
    
    override func moveUp(_ sender: Any?) {
        guard self.hasContent else {
            return
        }
        
        self.selectedCandidate = ((self.selectedCandidate ?? 0) - 1) %% self.tableView.numberOfRows
    }
    
    override func moveDown(_ sender: Any?) {
        guard self.hasContent else {
            return
        }
        
        self.selectedCandidate = ((self.selectedCandidate ?? -1) + 1) %% self.tableView.numberOfRows
    }
    
    override func mouseDown(with event: NSEvent) {
        let windowLocation = event.locationInWindow
        let tableViewLocation = self.tableView.convert(windowLocation, from: nil)
        
        let rowIndex = self.tableView.row(at: tableViewLocation)
        
        if rowIndex != -1 {
            self.selectedCandidate = rowIndex
        } else {
            super.mouseDown(with: event)
        }
    }
    
    // MARK: - NSTableViewDataSource
    func numberOfRows(in tableView: NSTableView) -> Int {
        return self.candidates?.count ?? 0
    }
    
    // MARK: - NSTableViewDelegate
    func tableView(_ tableView: NSTableView, viewFor tableColumn: NSTableColumn?, row: Int) -> NSView? {
        let view = tableView.makeView(withIdentifier: .candidateCell, owner: self) as? NSTableCellView
        
        view?.textField?.stringValue = self.candidates![row].text
        
        return view
    }
    
    func tableView(_ tableView: NSTableView, rowViewForRow row: Int) -> NSTableRowView? {
        // FIXME: does not react to window events
        class RowView: NSTableRowView {
            override var isEmphasized: Bool {
                get { return true }
                set {}
            }
        }
        
        return RowView()
    }
}
