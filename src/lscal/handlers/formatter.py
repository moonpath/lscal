from icalendar import Calendar

class Handler:
    """Handler class for formatted event information display with improved visual hierarchy"""
    
    def __call__(self, calendar: Calendar) -> None:
        print(f"\n{'='*80}")
        print(f"📅  CALENDAR EVENTS INSPECTION")
        print(f"{'='*80}\n")

        valid_types = ('VEVENT', 'VTODO', 'VJOURNAL', 'VFREEBUSY')
        
        count = 0
        for component in calendar.subcomponents:
            if component.name not in valid_types:
                continue
            
            count += 1
            summary = component.get('SUMMARY', 'No Title')
            
            print(f"[{count:02d}] {component.name} | {summary}")
            print(f"{'─' * 80}")

            for key, value in component.items():
                val_str = str(value).strip()
                print(f"  🔹 {key:<20} : {val_str}")

            if component.subcomponents:
                print(f"\n     🔽 Subcomponents ({len(component.subcomponents)})")
                
                for sub in component.subcomponents:
                    sub_name = sub.name

                    target_padding = 66 - len(str(sub_name))
                    padding_dashes = '─' * max(0, target_padding)
                    
                    print(f"     ┌── [ {sub_name} ] {padding_dashes}")
                    
                    for sub_key, sub_value in sub.items():
                        val_str = str(sub_value).strip()
                        print(f"     │   {sub_key:<16} : {val_str}")
                    
                    print(f"     └──{'─' * 72}")

            print(flush=True)
            
        if count == 0:
            print("No displayable components found.", flush=True)
